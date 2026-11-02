import llama_cpp
import torch, time, os, random, traceback, json,csv
from llama_cpp import Llama
from typing import List, Dict
from collections import defaultdict
import pandas as pd
from datetime import datetime, timedelta
from transformers import AutoTokenizer, AutoModelForCausalLM
from functools import partial
from argparse import ArgumentParser
from .utils_M import generate_sub_topics, generate_profiles, generate_full_monologue, evaluate_monologue_structure
from .utils_M import hf_phi_judge_M, hf_llama3_judge_M
from .utils_M import tokenizer, stop_tokens
from .utils_M import group_sample_monologues_by_dimension, select_samples_and_examples
from .utils_M import _clean_enums, make_profile_description, build_profiles_index
from .quality_metrics_M import  g_eval_M, calculate_metrics_M
from .config_M import (
    NUM_SUBTOPICS_PER_TOPIC, NUM_PROFILES, 
    MAX_MONOLOGUES_PER_USER, EXPLICIT_PROFILE_RATIO, SEED,MAX_USERS_PER_MONOLOGUE,MIN_USERS_PER_MONOLOGUE ,
    SAMPLE_MONOLOGUES_PATH, SAMPLES_PER_PROMPT, EXAMPLES_PER_DIMENSION, MONOLOGUE_GEN_MAX_RETRIES,
    topics_file_path, csv_path, cooloff_period, cooloff_frequency, Num_Monologues_Per_Subtopic,
    PROFILE_N_CTX,
    PROFILE_N_BATCH,
    PROFILE_N_UBATCH,
    PROFILE_N_THREADS,
    MONOLOGUE_N_CTX,
    MONOLOGUE_N_BATCH,
    MONOLOGUE_N_UBATCH,
    MONOLOGUE_N_THREADS, GPU_COOLDOWN_FREQUENCY ,GPU_COOLDOWN_SECONDS 
)


def clean_profile(p):
    age, gender, risk, tone, money, game = _clean_enums(
        p.get("age"),
        p.get("gender"),
        p.get("risk_level"),
        p.get("tone_baseline"),
        p.get("money_focus"),
        p.get("game_focus"),
    )
    p["age"] = age
    p["gender"]=gender
    p["risk_level"] = risk
    p["tone_baseline"] = tone
    p["money_focus"] = money
    p["game_focus"] = game

    # pass-through description (light sanitize inside make_profile_description)
    p["description"] = make_profile_description(p)
    return p


#start function for generating monologues

def pick_one_profile(profiles_for_subtopic, usage_map, subtopic):
    """Choose exactly one profile for this monologue.
    - Prefer the least-used user for this subtopic (round-robin fairness).
    - If none tracked yet, pick randomly.
    usage_map: dict[subtopic][user_id] -> count
    """
    import random
    if not profiles_for_subtopic:
        return None
    usage_map.setdefault(subtopic, {})
    # Count usage; default 0
    usage_pairs = []
    for p in profiles_for_subtopic:
        uid = p.get("user_id")
        usage_pairs.append((usage_map[subtopic].get(uid, 0), uid, p))
    # sort by usage then random tie-breaker
    random.shuffle(usage_pairs)
    usage_pairs.sort(key=lambda tup: tup[0])
    chosen_usage, chosen_uid, chosen_profile = usage_pairs[0]
    usage_map[subtopic][chosen_uid] = chosen_usage + 1
    return chosen_profile

#start function for generating monologues
def smg(csv_path:str, topics_file:str=None, n_sub_topics:int=None, n_personas:int=None, stage:str='scratch', 
        disable_dialog_quality_metrics: bool=False, disable_judge: bool=False, 
        cooloff_period: int=10, cooloff_frequency: int=20) -> pd.DataFrame:
    if stage not in ['scratch', 'metrics', 'llm_judge']:
        raise ValueError(f'invalid stage value: {stage}')
    
    
    print(f'\n ->Loading dimensioned sample monologues from {SAMPLE_MONOLOGUES_PATH}\n')
    monologues_grouped = group_sample_monologues_by_dimension(SAMPLE_MONOLOGUES_PATH)
    # light guard
    if sum(len(v) for v in monologues_grouped.values()) < 3:
        raise ValueError(" =>Not enough sample monologues found (need at least 3).")
  
    # activate this when generate topics / profiles
    
    if stage in ["scratch", "llm_judge"]:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        n_gpu_layers = 0 if device=="cuda" else 0 # server  later n_gpu_layers = 32
        gguf_model = Llama( #for server change model to Mistral
    model_path=os.path.join("models/Mistral", 'model.gguf'),  # adjust the filename and path as needed
    n_ctx=PROFILE_N_CTX, #8192,
    #n_batch=128,
    #n_ubatch=64,
    verbose=False,
    n_gpu_layers=n_gpu_layers,
    n_threads=2  # Optional: use appropriate number of monologues for CPU
        )

    else:
        gguf_model = None   
    
    #print("After loading model", flush=True)    #server - remove later
    
    gambling_topics = []
    other_topics = []
    current_section = None
    if stage=='scratch':
        if not os.path.exists(topics_file):
            raise OSError(f"{topics_file} does not exist")
        # read the topics.txt file
        with open(topics_file, "r", encoding="utf-8") as f:           
            for line in f:
                topic = line.strip()
                if not topic or topic.startswith("#"):
                    if "gambling" in topic.lower():
                        current_section = "gambling"
                    elif "other" in topic.lower():
                        current_section = "other"
                    continue

                if current_section == "gambling":
                    gambling_topics.append(topic)
                elif current_section == "other":
                    other_topics.append(topic)
             
        monologues_df = pd.DataFrame(columns=['monologue_id', 'main_user', 'main_post','main_timestamp'])
        
    # wipe previous artifacts at the start of a fresh run
        '''
        for p in ["1_Generated_Subtopics_M.csv", "2_User_profiles_M.csv",
          "2_User_profiles_extended_M.csv", "Participation_Matrix_M.csv",
          "3_Generated_monologues.jsonl", "3_Generated_monologues.csv",
          "used_dimensions_log_M.txt"]:
         if os.path.exists(p):
             os.remove(p)
        '''
#======================================================================================
    ##step 1: Generating s subtopics from t topics in topics.txt
        all_sub_topics = dict()
        
        for base_topic in gambling_topics+other_topics:
            try: 
                print(f' Step 1->Generating subtopics for topic: {base_topic}')
                sub_topics = generate_sub_topics(model=gguf_model, topic=base_topic, n_sub_topics=n_sub_topics or NUM_SUBTOPICS_PER_TOPIC)
                all_sub_topics[base_topic] = sub_topics
                print(f" ->{len(sub_topics)} subtopics have been generated\n")# for sub topic {base_topic}")
            except Exception as e:
             print(f" Failed to process topic {base_topic}")
             with open("generation_failures.log", "a", encoding="utf-8") as f:
                f.write(f"Subtopic generation error for topic: {base_topic}, reason: {e}\n")
        
        print("\033[93m => Generating subtopics completed\033[0m\n\n")
        '''
        
        # NEW === Read subtopics from CSV instead of generating ===
        subtopics_path = "Path\\1_Generated_Subtopics.csv"

        subtopic_df = pd.read_csv(subtopics_path)
        # Build the dict structure expected by build_participation_matrix:
        # { "<Topic>": ["<Subtopic 1>", "<Subtopic 2>", ...], ... }
        all_sub_topics = (
            subtopic_df
            .groupby("Topic")["Subtopic"]
            .apply(list)
            .to_dict()
        )

        print(f"Prepared {sum(len(v) for v in all_sub_topics.values())} subtopics "
            f"across {len(all_sub_topics)} topics for the participation matrix.")

        # Optional: lowercase column names for robustness
        subtopic_df.columns = [col.lower() for col in subtopic_df.columns]

        # Convert to list of dicts as expected by downstream code
        subtopics = [
            {"topic": row["topic"], "subtopic": row["subtopic"]}
            for _, row in subtopic_df.iterrows()
        ]

        print(f"\033[93m => Loaded {len(subtopics)} subtopics from {subtopics_path}\033[0m\n\n")
        
        '''     
#======================================================================================     
        #step 2: Generating p user profiles
        
        try:       
            print(f' Step 2->Generating profiles for all subtopics\n')

            # Flatten the topic list from all_sub_topics
            topic_list = gambling_topics+other_topics#[topic for topic in all_sub_topics.keys()]
    
            # Generate P global profiles with randomized interest_areas
            profiles = generate_profiles(model=gguf_model,  n_profiles=NUM_PROFILES, explicit_ratio=EXPLICIT_PROFILE_RATIO)
    
            print(f" -> {len(profiles)} user profiles generated across all topics\n")
        except Exception as e:
            print(f" Failed to generate user profiles: {e}")
            print(f" Failed to generate user profiles: {e}", flush=True)#-
            traceback.print_exc()#-
            raise#-
            with open("generation_failures.log", "a", encoding="utf-8") as f:
                f.write(f"Global profile generation error: {e}\n")
                   
        print("\033[93m \n => Generating profiles completed\033[0m\n\n")

        #print("After generate_profiles", flush=True)  # sever - remove later
        '''
        # === Step 2: Load profiles from CSV instead of regenerating ===
        profiles_csv = "2_User_profiles_extended.csv"  # adjust if needed

        if not os.path.exists(profiles_csv):
            raise FileNotFoundError(f"{profiles_csv} not found. Generate profiles once, then reuse.")

        profiles_df = pd.read_csv(profiles_csv)

        # Basic sanity: required columns (clean_profile/_clean_enums use these)
        '''
        required = {
            "user_id","age","gender","risk_level","tone_baseline",
            "money_focus","brand_focus", "game_focus", "emotional_profile","behavior_traits","description", "personality_traits"
        }
        missing = required - set(c.lower() for c in profiles_df.columns)
        if missing:
            # try case-insensitive align
            cols_map = {c.lower(): c for c in profiles_df.columns}
            if missing - set(cols_map.keys()):
                raise ValueError(f"Profiles CSV missing required columns: {sorted(missing)}")
            profiles_df = profiles_df.rename(columns={cols_map[k]: k for k in required if k in cols_map})
        '''

        required = {
            "user_id",
            "age",
            "gender",
            "risk_level",
            "tone_baseline",
            "money_focus",
            "brand_focus",
            "game_focus",
            "emotional_profile",
            "behavior_traits",
            "description",
            "personality_traits",
        }

        column_map = {
            str(column).strip().lower(): column
            for column in profiles_df.columns
        }

        missing = required - set(column_map)

        if missing:
            raise ValueError(
                f"Profiles CSV missing required columns: {sorted(missing)}"
            )

        profiles_df = profiles_df.rename(
            columns={
                original_name: normalized_name
                for normalized_name, original_name in column_map.items()
                if normalized_name in required
            }
        )

        # Deduplicate/clean minimal
        profiles_df["user_id"] = profiles_df["user_id"].astype(str).str.strip()
        profiles_df = profiles_df.dropna(subset=["user_id"]).drop_duplicates(subset=["user_id"])

        # Convert to list[dict] for downstream code paths that expect it
        profiles = profiles_df.fillna("").to_dict(orient="records")

        print(f"Loaded {len(profiles)} profiles from {profiles_csv}")
        print("\033[93m \n => Loading profiles completed\033[0m\n\n")
        print("Sleeping 5 seconds")
        time.sleep(5)
        '''
#======================================================================================    
        #step 2.5 Generate Participation Matrix based on profiles, topics and subtopics 
        profiles_df = pd.DataFrame(profiles)

        #create the matrix
        
        matrix_df = build_participation_matrix( #matrix_df is never used as function does not return values
    all_sub_topics=all_sub_topics,
    profiles_df=profiles_df,
    save_path="Participation_Matrix_M.csv",
    min_users_per_subtopic=MIN_USERS_PER_MONOLOGUE,
    max_users_per_subtopic=MAX_USERS_PER_MONOLOGUE,
    max_monologues_per_user=MAX_MONOLOGUES_PER_USER,
    seed=SEED,
    verbose=True,
)
   
 
        
#======================================================================================
        #step 3 Generating monologues
        #SWITCHING model from Mistral to Llama 
        if stage in ["scratch", "llm_judge"]:
            '''
            device = "cuda" if torch.cuda.is_available() else "cpu"
            #n_gpu_layers = -1 if device=="cuda" else 0 # server  later n_gpu_layers = 32
            if device == "cuda" and not llama_cpp.llama_supports_gpu_offload():
                raise RuntimeError(
                    "CUDA is visible to PyTorch, but llama-cpp-python "
                    "does not support GPU offloading. Activate myenv_gpu."
                )

            n_gpu_layers = -1 if device == "cuda" else 0
            '''
            gpu_offload_available = llama_cpp.llama_supports_gpu_offload()

            if not gpu_offload_available:
                raise RuntimeError(
                    "llama-cpp-python does not support GPU offloading. "
                    "Activate myenv_gpu."
                )

            n_gpu_layers = -1

            print(
                "Creating monologue model with full CUDA offload: "
                f"n_gpu_layers={n_gpu_layers}",
                flush=True,
            )


            print("creting model first time")
            gguf_model = Llama(model_path=os.path.join("models/Llama3", 'model.gguf'), n_ctx=MONOLOGUE_N_CTX, #4096
                                n_batch=MONOLOGUE_N_BATCH,
                               n_ubatch=MONOLOGUE_N_UBATCH,
                                verbose=False, n_gpu_layers=n_gpu_layers, n_threads=MONOLOGUE_N_THREADS )
        else:
            gguf_model = None 
              

        # Load Participation Matrix
        participation_df = pd.read_csv("Participation_Matrix_M.csv", index_col=0)

        # Map user_id to full profile
        user_id_to_profile = {p["user_id"]: p for p in profiles}

        # Build subtopic -> list of user profiles
        sub_topic_profiles = defaultdict(list)
        for subtopic in participation_df.index:
            active_users = participation_df.columns[participation_df.loc[subtopic] == 1].tolist()

            # Clean and shape the profiles correctly before monologue generation
            cleaned_profiles = [clean_profile(user_id_to_profile[uid]) for uid in active_users if uid in user_id_to_profile]
            sub_topic_profiles[subtopic] = cleaned_profiles

        all_sub_topics_profiles = sub_topic_profiles
        #monologue_counter=1
        #output_file = "3_Generated_monologues.jsonl"
        #open(output_file, "w").close()
        if os.path.exists("3_Generated_monologues.csv"):
            os.remove("3_Generated_monologues.csv")
        #all_monologues_list = []
        generation_attempts = 0
        profiles_index = build_profiles_index(profiles_df)
        for sub_topic, user_profiles in sub_topic_profiles.items(): 
            base_topic = sub_topic.split(" - ")[0]  # get parent topic
            examples_block = ""
            if base_topic in gambling_topics:
                # Apply gambling-aware prompt with dimension constraints
                # choose 3 dimensions; 1 sample monologue per chosen dimension; and 3 cues each
                sample_monologues, dimension_instructions, examples_block, chosen_short_dims,reflection_area,reflection_block = \
                select_samples_and_examples(
                grouped=monologues_grouped,
                k_dims=SAMPLES_PER_PROMPT,                 # == 3
                examples_per_dim=EXAMPLES_PER_DIMENSION   # == 3
                )
                '''
                # Log which short-dimension sets were injected
                with open("used_dimensions_log_M.txt", "a", encoding="utf-8") as f:
                    f.write(f"{sub_topic} | {', '.join(chosen_short_dims)}\n")
                '''


                prompt_type = "addiction"
            else:
                #  Use standard social media prompt but still feed style samples
                dimension_instructions = ""
                examples_block = ""
                reflection_area = None
                reflection_block = ""
                prompt_type = "generic"

                # flatten all grouped monologues and sample 3
                flat_samples = [rec for arr in monologues_grouped.values() for rec in arr]
                if len(flat_samples) < 3:
                    raise ValueError("Not enough sample monologues loaded to seed generic prompts.")
                sample_monologues, chosen_short_dims = random.sample(flat_samples, 3), []
            
            
            if not user_profiles:
                print(f" => No profiles exist for subtopic {sub_topic}, skipping monologue generation.")
                continue
            try:
                # For each active user in this subtopic, generate one monologue
                log_file = "used_dimensions_log_M.csv"

                if not os.path.exists(log_file):
                    with open(log_file, "w", newline="", encoding="utf-8-sig") as f:
                        writer = csv.writer(f)
                        writer.writerow([
                            "monologue_id",
                            "user_id",
                            "risk_level",
                            "topic",
                            "subtopic",
                            "reflection_area",
                            "prompt_type",
                            "ogdq_dim_1",
                            "ogdq_dim_2",
                            "ogdq_dim_3",
                            "negative_urgency",
                            "positive_urgency",
                            "lack_of_premeditation",
                            "lack_of_perseverance",
                            "sensation_seeking"
                        ])
                
                for u_idx, profile in enumerate(user_profiles, start=1):
                    # Randomize a start date for the monologue within 2025
                    days_offset = random.randint(0, 364)
                    start_time = datetime(2025, 1, 1) + timedelta(days=days_offset)

                    uid = profile.get("user_id", f"UXXXX")
                    monologue_id = f"{sub_topic.replace(' ', '_')}_{uid}"
                    #traits = json.loads(profile["personality_traits"])
                    #labels = traits["labels"]
                    try:
                        traits = json.loads(profile.get("personality_traits", "{}"))
                        labels = traits.get("labels", {})
                    except (TypeError, json.JSONDecodeError):
                        print(
                            f"Invalid personality_traits for user {uid}; skipping profile.",
                            flush=True,
                        )
                        continue

                    required_trait_labels = {
                        "negative_urgency",
                        "positive_urgency",
                        "lack_of_premeditation",
                        "lack_of_perseverance",
                        "sensation_seeking",
                    }

                    missing_trait_labels = required_trait_labels - labels.keys()

                    if missing_trait_labels:
                        print(
                            f"Missing UPPS-P labels for user {uid}: "
                            f"{sorted(missing_trait_labels)}; skipping profile.",
                            flush=True,
                        )
                        continue
                    
                    #monologue_id = f"{sub_topic.replace(' ', '_')}_{uid}"
                    print(f'\n Step 3->Generating monologue for subtopic "{sub_topic}" as user {uid}')

                    max_retries = MONOLOGUE_GEN_MAX_RETRIES
                    retry_count = 0
                    approved = False
                    #print("Before while loop in step 3")
                    while retry_count < max_retries and not approved:
                        #print("Sleeping 5 seconds")
                        #time.sleep(5)

                        # Generate monologue for THIS user only
                        '''
                        if(monologue_counter%3==0):
                            print("deleting model")
                            del gguf_model
                            gc.collect()
                            print("recreating model again")
                            gguf_model = Llama(
                                model_path=os.path.join("models/Llama3", "model.gguf"),
                                n_ctx=MONOLOGUE_N_CTX,
                                n_batch=MONOLOGUE_N_BATCH,
                                n_ubatch=MONOLOGUE_N_UBATCH,
                                verbose=False,
                                n_gpu_layers=n_gpu_layers,
                                n_threads=MONOLOGUE_N_THREADS,
                            )
                        '''
                        #monologue_counter+=1

                        #print("calling generate_full_monologue", flush=True)
                        monologue = generate_full_monologue(
                            model=gguf_model,
                            subtopic=sub_topic,
                            user_profile=profile,                   # <— single specific user
                            monologue_id=monologue_id,
                            sample_monologues=sample_monologues,
                            start_time=start_time,
                            dimension_instructions=dimension_instructions,
                            prompt_type=prompt_type,
                            examples_block=examples_block,
                            reflection_area=reflection_area,
                            reflection_block=reflection_block
                        )
                        generation_attempts += 1

                        if (
                            GPU_COOLDOWN_FREQUENCY > 0
                            and generation_attempts % GPU_COOLDOWN_FREQUENCY == 0
                        ):
                            print(
                                f"\nCooling pause after {generation_attempts} generation attempts. "
                                f"Sleeping for {GPU_COOLDOWN_SECONDS} seconds...\n",
                                flush=True,
                            )
                            time.sleep(GPU_COOLDOWN_SECONDS)
                        if monologue is None:
                            print(" => generate_full_monologue returned None, retrying...")
                            retry_count += 1
                            continue

                        allowed = {uid}  # <— validate only this user appears
                        
                        #profiles_index = build_profiles_index(profiles_df)
                        #print("calling evaluate_monologue_structure")
                        issues = evaluate_monologue_structure(
                            monologue,
                            allowed_users=allowed,
                            profiles_index=profiles_index,  # <-- new argument
                        )

                        if not issues:
                            approved = True
                            #print("open log_file")
                            with open(log_file, "a", newline="", encoding="utf-8-sig") as f:
                                writer = csv.writer(f)
                                writer.writerow([
                                    monologue_id,
                                    uid,
                                    profile["risk_level"],
                                    base_topic,
                                    sub_topic,
                                    reflection_area,
                                    prompt_type,
                                    chosen_short_dims[0] if len(chosen_short_dims) > 0 else "",
                                    chosen_short_dims[1] if len(chosen_short_dims) > 1 else "",
                                    chosen_short_dims[2] if len(chosen_short_dims) > 2 else "",
                                    labels["negative_urgency"],
                                    labels["positive_urgency"],
                                    labels["lack_of_premeditation"],
                                    labels["lack_of_perseverance"],
                                    labels["sensation_seeking"],    
                                ])
                            #with open(output_file, "a", encoding="utf-8") as f:
                                #f.write(json.dumps(monologue, ensure_ascii=False) + "\n")
                                #print(f'monologue saved in {output_file} for subtopic: {sub_topic} (user {uid})')

                            monologue_csv_row = {
                                "monologue_id": monologue["monologue_id"],
                                "topic": sub_topic,
                                "main_user": monologue["main_post"]["user_id"],
                                "main_post": monologue["main_post"]["text"],
                                "main_timestamp": monologue["main_post"]["timestamp"]
                            }

                            #print("dataframe to  monologues.csv file")
                            pd.DataFrame([monologue_csv_row]).to_csv(
                                "3_Generated_monologues.csv",
                                mode="a",
                                index=False,
                                header=not os.path.exists("3_Generated_monologues.csv"),
                                quoting=csv.QUOTE_ALL,
                                encoding='utf-8-sig',
                                lineterminator="\n"
                            )
                            print(f'monologue saved in 3_Generated_monologues.csv for {sub_topic} (user {uid})')
                            
                            
                            #all_monologues_list.append(monologue_csv_row)

                        else:
                            print(f" Issues in monologue {monologue_id}: {issues}")
                            retry_count += 1
                            

                    if not approved:
                        print(f" Could not get a valid monologue for {monologue_id} after {max_retries} attempts.")
                        

                # Optional cooloff per subtopic, unchanged:
                #if cooloff_frequency and ((list(all_sub_topics_profiles.keys()).index(sub_topic) + 1) % cooloff_frequency == 0):
                    #print(f"Cooling off for {cooloff_period} seconds...")
                    #time.sleep(cooloff_period)

            except Exception as e:
                print(f" Failed to generate monologue: {e}")
               
                traceback.print_exc()
                with open("generation_failures.log", "a", encoding="utf-8") as f:
                    f.write(f"monologue generation fail for subtopic: {sub_topic}, reason: {e}\n")

            
        #monologues_df = pd.DataFrame(all_monologues_list)
        
        print("\033[93m \n => Generating monologues completed\033[0m\n\n")
    """
    # === Instead of building DataFrame from in-memory list, read from saved CSV ===
    csv_input_path = csv_path #"3_Generated_monologues.csv"
    if not os.path.exists(csv_input_path):
            raise FileNotFoundError(f"{csv_input_path} not found - make sure monologue generation completed and file was saved.")

    print(f"\n ->Loading previously generated monologues from {csv_input_path}\n")
    monologues_df = pd.read_csv(
            csv_input_path,
            dtype_backend="pyarrow",     # efficient Arrow-backed strings (less RAM)
            engine="pyarrow"
        )

    print(f"Loaded {len(monologues_df)} monologues from {csv_input_path}")
    print("\033[93m \n => Generating monologues completed (loaded from CSV)\033[0m\n\n")
    """
    
    '''
    #Metrics (Judge & G-Eval) using HF Phi-3 & Llama-3

    # === HF Llama-3 judge loader (fast) ===

    llama_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
    llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_id)
    if llama_tokenizer.pad_token_id is None:
        llama_tokenizer.pad_token = llama_tokenizer.eos_token

    llama_model = AutoModelForCausalLM.from_pretrained(
        llama_model_id,
        torch_dtype=torch.float16,   # or torch.bfloat16 if your GPU prefers
        device_map="auto",
        low_cpu_mem_usage=True
    )
    # hygiene to silence PAD messages
    llama_model.config.pad_token_id = llama_tokenizer.pad_token_id
    if hasattr(llama_model, "generation_config"):
        llama_model.generation_config.pad_token_id = llama_tokenizer.pad_token_id

    # Try to keep fast cache; if your transformers is older, this line is harmless
    try:
        llama_model.generation_config.cache_implementation = "static"
    except Exception:
        pass

    # === Run the judge (prefix-clamped) ===


    if stage in ["scratch", "llm_judge"] and not disable_judge:
        print(' Step 4->Calculating judge scores with HF Llama-3...')
        torch.cuda.empty_cache()
        monologues_df = hf_llama3_judge_M(
            model=llama_model,
            tokenizer=llama_tokenizer,
            df=monologues_df,
            max_new_tokens=4,         # tiny decode -> fast
            cooldown_every=200,
            cooldown_seconds=5
        )
        monologues_df.to_csv(csv_path, index=False, quoting=csv.QUOTE_ALL,
                            encoding='utf-8-sig', lineterminator="\n")
        print(f'\033[93m \n => monologues saved with HF Llama-3 judge scores.\033[0m\n\n')

    
    # generate G-Eval metrics using Llama3 from HF
    if stage in ["scratch", "metrics", "llm_judge"]:
        if not disable_dialog_quality_metrics:
               
                torch.cuda.empty_cache()
                print(" >>>>>> cooling off for: 5 seconds")
                time.sleep(5)

                print(' Step 5->Calculating G-Eval metrics for the generated monologues by Llama-3...')
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                
                #Llama3
                model_id = "meta-llama/Meta-Llama-3-8B-Instruct"  # accept license on HF
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                model = AutoModelForCausalLM.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16,
                    device_map="auto",
                )

                metric_funcs = {
                    'g_eval_llama': partial(g_eval_M, model=model, tokenizer=tokenizer)
                }
                metric_scores = calculate_metrics_M(sentences=monologues_df['main_post'].tolist(), metric_funcs=metric_funcs)
                for metric_name, scores in metric_scores.items(): 
                    monologues_df[metric_name] = scores
                    print(f'\t ->Metric {metric_name} calculated...')

                monologues_df.to_csv(csv_path, index=False,  quoting=csv.QUOTE_ALL,
                                        encoding='utf-8-sig',
                                        lineterminator="\n")
                print(f'\033[93m \n =>All the generated monologues with G-Eval metrics calculated using Llama-3, have been saved\033[0m\n\n')# to {csv_path}')

    ## llm-as-a-judge scores using Phi3 from HF

    model_id = "microsoft/Phi-3.5-mini-instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    # Make sure PAD is defined
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,      # or torch.float16 if your GPU prefers that
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    # Silence PAD warnings globally
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.do_sample = False
    
    # NEW: if judge is enabled, do it here with the HF model
    if stage in ["scratch", "llm_judge"] and not disable_judge:
        print(' Step 4->Calculating judge scores with HF Phi-3...')
        # NEW
        monologues_df = hf_phi_judge_M(model, tokenizer, monologues_df)
        monologues_df.to_csv(csv_path, index=False, quoting=csv.QUOTE_ALL,
                            encoding='utf-8-sig', lineterminator="\n")
        print(f'\033[93m \n =>All monologues saved with HF Phi-3 judge scores.\033[0m\n\n')

        
    
    # generate G-Eval metrics using Phi3 from HF
    if stage in ["scratch", "metrics", "llm_judge"]:
        if not disable_dialog_quality_metrics:
               
                torch.cuda.empty_cache()
                print(" >>>>>> cooling off for: 5 seconds")
                time.sleep(5)

                print(' Step 5->Calculating G-Eval metrics for the generated monologues by Phi-3...')
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                #Phi3
                tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct", trust_remote_code=True)
                model = AutoModelForCausalLM.from_pretrained("microsoft/Phi-3.5-mini-instruct", torch_dtype=torch.bfloat16, trust_remote_code=True).to(device)
                
                metric_funcs = {
                    'g_eval_phi': partial(g_eval_M, model=model, tokenizer=tokenizer)
                }
                metric_scores = calculate_metrics_M(sentences=monologues_df['main_post'].tolist(), metric_funcs=metric_funcs)
                for metric_name, scores in metric_scores.items(): 
                    monologues_df[metric_name] = scores
                    print(f'\t ->Metric {metric_name} calculated...')

                monologues_df.to_csv(csv_path, index=False,  quoting=csv.QUOTE_ALL,
                                        encoding='utf-8-sig',
                                        lineterminator="\n")
                print(f'\033[93m \n =>All the generated monologues with G-Eval metrics calculated using Phi-3, have been saved\033[0m\n\n')# to {csv_path}')
   '''                
    



def build_participation_matrix(
    all_sub_topics: dict,
    profiles_df: pd.DataFrame,
    save_path: str = "Participation_Matrix_M.csv",
    min_users_per_subtopic: int = MIN_USERS_PER_MONOLOGUE,
    max_users_per_subtopic: int = MAX_USERS_PER_MONOLOGUE,
    max_monologues_per_user: int = MAX_MONOLOGUES_PER_USER,
    seed: int = 42,
    verbose: bool = True,
):

    random.seed(seed)
    Num_monologues=0
    # ---- flatten subtopics and keep lookup ----
    full_subtopics = []
    topic_lookup = {}
    for topic, subs in all_sub_topics.items():
        for s in subs:
            full = f"{topic} - {s}"
            full_subtopics.append(full)
            topic_lookup[full] = topic

    profiles_df = profiles_df.copy()

    user_ids = profiles_df["user_id"].tolist()
    usage = {uid: 0 for uid in user_ids}

    # Helper to sample k participants with low-usage bias
    def pick_participants(topic_name: str, k: int) -> list[str]:
        backfill_sorted = sorted(user_ids, key=lambda u: usage[u])
        chosen = []
        
        for u in backfill_sorted:
            if usage[u] < max_monologues_per_user:
                chosen.append(u)
                if len(chosen) == k: break
        if len(chosen) < k:
            remaining = [u for u in user_ids if usage[u] < max_monologues_per_user and u not in chosen]
            random.shuffle(remaining)
            chosen += remaining[: k - len(chosen)]
        random.shuffle(chosen)
        return chosen

    subtopic_to_users = {}
    for full in full_subtopics:
        k = random.randint(min_users_per_subtopic, max_users_per_subtopic)
        users = pick_participants(topic_lookup[full], k)
        subtopic_to_users[full] = users
        for u in users: usage[u] += 1

    # Ensure no two subtopics have the exact same participant set
    seen_sets: dict[tuple[str, ...], str] = {}
    for st, users in list(subtopic_to_users.items()):
        key = tuple(sorted(users))
        if key in seen_sets:
            # mutate: swap last member with a low-usage alternative not already in the set
            alternatives = sorted(
                [u for u in user_ids if u not in users and usage[u] < max_monologues_per_user],
                key=lambda u: usage[u],
            )
            if alternatives:
                replaced = users[-1]
                new_u = alternatives[0]
                users[-1] = new_u
                usage[replaced] = max(0, usage[replaced] - 1)
                usage[new_u] += 1
                subtopic_to_users[st] = users
                key = tuple(sorted(users))
        seen_sets[key] = st

    # Create matrix DataFrame
    matrix = pd.DataFrame(0, index=full_subtopics, columns=user_ids)
    for st, lst in subtopic_to_users.items():
        for uid in lst:
            matrix.at[st, uid] = 1
            Num_monologues+=1

    # Save and return
    matrix.index.name = "subtopic"
    matrix.to_csv(save_path)
    if verbose:
        print(f"Participation matrix saved to {save_path} "
              f"({len(full_subtopics)} subtopics X {len(user_ids)} users)")

    print("\033[93m \n => Generating matrix completed\033[0m\n")
    print(f"\033[93m \n => Total number of monologues expected = {Num_monologues}\033[0m\n\n")

    return matrix




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--stage', type=str, default='scratch')
    parser.add_argument('--disable_judge', default=False, action="store_true")
    parser.add_argument('--disable_dialog_quality_metrics', default=False, action="store_true")#was False
    args = parser.parse_args()
    with open(topics_file_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    num_topics = len(lines)
try: 
    print(" ======================================================")
    print(" Statistics based on inputs:")
    print(f" Number of topics = {num_topics}")
    print(f" Number of sub topics per topic = {NUM_SUBTOPICS_PER_TOPIC}")
    print(f" Total number of subtopics = {num_topics*NUM_SUBTOPICS_PER_TOPIC}")
    print(f" Total number of user profiles = {NUM_PROFILES}")
    print(" ======================================================")

    smg(        
        topics_file=topics_file_path, n_sub_topics=NUM_SUBTOPICS_PER_TOPIC,
        stage=args.stage, csv_path=csv_path, n_personas=NUM_PROFILES,
        disable_judge=args.disable_judge, disable_dialog_quality_metrics=args.disable_dialog_quality_metrics, 
        cooloff_period=cooloff_period, cooloff_frequency=cooloff_frequency)
  
    print(f'\033[93m -> Execution Completed! Data saved successfully to {csv_path}\033[0m\n\n')
except Exception as e:
    print(f"Error {e}")