from .config_M import OPENER_MAX_WORDS,  EMOJI_MIN, EMOJI_MAX, OPENER_MIN_WORDS

generate_topics_prompt = """
You are an expert topic generator. Generate exactly {n_sub_topics} unique, niche subtopics related to the given main topic.
These subtopics should reflect the types of real-life concerns, experiences, or questions that everyday users would discuss informally on Twitter.

Main topic: "{topic}"

Instructions:
Output only a valid Python list of {n_sub_topics} strings.
Return ONLY a valid Python list. Your response must begin with "[" and end with "]".
Do not include any explanations or comments. 
Avoid formal phrasing or academic terms.
Ensure subtopics are short (≤10 words), raw, and sound like real tweets — use casual grammar, contractions, and emotional tone (e.g., "idk", "felt like", "tired of", etc.).
No links, hashtags, or @-mentions.
Avoid words like "addiction", "relapse", "disorder", "therapy" or "recovery".
Do not directly mention medical, legal, or diagnostic terms.
Subtopics should be emotionally expressive and realistic, spanning positive, negative and mixed moments.
Include everyday wins, near-misses, brief self-control, plus moments of discomfort or risky pulls—but don't stack multiple heavy emotions in one line.
Across the whole batch, vary valence to roughly 30-35% non-negative, 30-35% mixed, 30-35% negative. Do not default to neutral/negative.
Within each list of {n_sub_topics} subtopics, include ≥1 non-negative and ≥1 mixed/negative subtopic. If not satisfied, regenerate the list.  
Avoid clinical tone or confessions - focus on behavior and emotions, not labels.
Include a mix of drivers such as self-doubt, emotional tension, inner conflict, peer influence, social consequences, curiosity, restraint, relief, small wins, and walk-aways. Don't cram multiple heavy emotions into a single subtopic.
Ensure all subtopics are unique and written in natural, informal language.
Vary tone across subtopics: frustration, sarcasm, denial, regret, joy, sadness, numbness, and also hope, relief, calm, pride, gratitude, playful banter, matter-of-fact. Do not default to negative outcomes.
Favor phrasing that captures how people feel or justify behaviors, not direct actions.
Make sure the list is syntactically correct and fully enclosed in square brackets [].

Use the following examples as a reference:

Example 1:
topic: Gambling Urges
n_sub_topics: 4
Response:
[
  "Edgy all afternoon; calmed once kickoff started",
  "Small win; closed the app, felt lighter",
  "Muted the picks chat; cooked pasta instead",
  "2am scrolling again; promised myself, tomorrow chill"
]

Example 2:
topic: Chasing Losses
n_sub_topics: 3
Response:
[
  "Tried to win back yesterday; made it worse",
  "Stopped mid-chase; walked home a bit steadier",
  "Even winning felt empty; wanted one more"
]

Example 3:
topic: Stress and Mood
n_sub_topics: 3
Response:
[
  "Foggy morning; tiny wins felt bigger",
  "Work stacked; chest tight by lunch",
  "Skipped plans to recharge; guilty, relieved"
]
"""
# ---------------------------------------------------------------------------

generate_user_profiles_prompt = """
You are an expert in creating Twitter-style user profiles for a synthetic dataset.
Generate exactly {n_profiles} diverse, highly specific user profiles as PIPE-delimited rows.

Required 12 columns (in order, separated by " | "):
user_id | age | gender | risk_level | tone_baseline | money_focus | brand_focus | game_focus | emotions | behaviors | description | personality_traits

Definitions & constraints:
- user_id: U0001, U0002, ... (increment sequentially; keep 4 digits)
- age: must be an integer between 18 and 70 inclusive.
- gender: male or female.
- risk_level: one of {{none, low, moderate, high}} to represent level of gambling risk. Include all four risk levels across the batch.
Keep the distribution approximately balanced whenever possible.
- tone_baseline: one or two tones chosen from {{angry, sarcastic, ashamed, defensive, hopeful, bitter, anxious, jaded, helpful, analytical, relieved, steady, calm, proud}}
  - Batch valence target: ~40% negative-toned (angry/sarcastic/ashamed/defensive/bitter/anxious/jaded), ~20% neutral/positive-toned (analytical/hopeful/helpful/relieved/steady/calm/proud), and ~40% profiles whose description conveys mixed feelings (both good and hard notes). Do not use "mixed" as tone_baseline.
- money_focus: one of {{stakes_small, stakes_mid, stakes_high}}
  - stakes_small ≈ <€20; stakes_mid ≈ €20-120; stakes_high ≈ >€120
- brand_focus: pick one brand from the list {{888, Stake, Bet365, Betfair, WilliamHill, PaddyPower, Unibet, LuckySpin, QuickBet, SlotKing}}
- game_focus: pick one from {{sportsbook, slots, live_casino, blackjack, roulette, poker, crash, lootboxes, scratchers}}
- emotions: two single words, comma-separated (e.g., "bitter, anxious, frustrated"). Single tokens only (no spaces or hyphens).
- behaviors: two short phrases, comma-separated (e.g., "late-night ATM, blames RNG, maxed card")
- description: 1-2 casual short sentences that imply habits/feelings/behavior. Normal gambling context is allowed, such as betting, casino, slots, sportsbook, poker, roulette, brands, money, or late-night play. However, avoid clinical/diagnostic labels and extreme life-collapse phrasing. Forbidden description terms: "addiction", "addicted", "gambling addiction", "problem gambling", "therapy", "diagnosis", "relapse", "disorder", "recovery journey", "ruined my life", "lost everything", "hit rock bottom".
- personality_traits: JSON with UPPS-P scores on a 1.0-4.0 scale.
  The five UPPS-P dimensions are:
  negative_urgency, positive_urgency, lack_of_premeditation, lack_of_perseverance, sensation_seeking.
  Higher scores indicate higher impulsivity for that dimension. e.g. low = 1.00-1.99, mid = 2.00-2.99, high = 3.00-4.00.
  UPPS-P interpretation:
- Negative Urgency:
  tendency to act impulsively during negative emotions
  (stress, frustration, sadness, anger)

- Positive Urgency:
  tendency to act impulsively during positive emotions
  (excitement, celebration, confidence)

- Lack of Premeditation:
  tendency to act without considering consequences

- Lack of Perseverance:
  difficulty staying focused on boring or difficult tasks

- Sensation Seeking:
  tendency to seek excitement, novelty, stimulation and risk

  Example:
  {{"model":"UPPS-P","values":{{"negative_urgency":3.7,"positive_urgency":2.8,"lack_of_premeditation":3.5,"lack_of_perseverance":1.8,"sensation_seeking":3.9}}}}
- Make sure profiles are reasonably balanced across low, mid, and high levels for all five UPPS-P dimensions.
- The personality_traits field must contain valid JSON only.
- All UPPS-P values must be floating-point numbers between 1.0 and 4.0 inclusive.
- Generate realistic, diverse, and psychologically plausible UPPS-P profiles.
- The five dimensions are not completely independent.
- Plausible tendencies include:
High Negative Urgency is somewhat more likely to co-occur with High Positive Urgency.
High Lack of Premeditation is somewhat more likely to co-occur with High Lack of Perseverance.
High Sensation Seeking is somewhat more likely to co-occur with High Positive Urgency.

- These are tendencies only.
- Do not make them deterministic.
- Maintain diversity.

-Risk level and UPPS-P should be psychologically plausible.

NONE:
- mostly low impulsivity
- at most one high dimension
- casual, occasional, or purely observational gambling-related activity only
- no financial stress, chasing losses, secrecy, or loss-of-control themes
- no repeated gambling behavior as a central life activity
- behaviors should be supportive, informational, social, or low-risk
- descriptions should not imply gambling-related problems

LOW:
- mixture of low and mid dimensions
- occasional high dimension allowed
- gambling is present but generally controlled
- minor frustration, temptation, or occasional over-involvement is acceptable
- no severe consequences

MODERATE:
- multiple mid dimensions
- one or two high dimensions common
- some recurring gambling involvement
- occasional chasing, regret, secrecy, or difficulty sticking to limits may appear
- moderate emotional or financial impact is acceptable

HIGH:
- usually high Negative Urgency and/or high Lack of Premeditation
- often elevated Lack of Perseverance
- may also show high Sensation Seeking
- gambling is a recurring and important activity
- descriptions may imply chasing losses, repeated play, secrecy, emotional distress, or difficulty controlling behavior
- financial concerns may appear, but avoid extreme life-collapse stories

Ensure the description, emotions, behaviors, money_focus, and risk_level are mutually consistent.

The description, emotions, and behaviors must also be consistent with both the risk_level and the UPPS-P profile, but do not mention UPPS-P trait names in the description.
Ensure UPPS-P traits and risk level are consistent.

Style:
- Safety/formatting: No links, hashtags, @-mentions, nor the "|" character in any field.
- Make every profile distinct in age, life context, attitude, tone, and habits.
- Vary money_focus, brand_focus, game_focus, emotions and behaviors widely across the set.
- Keep language informal and social-media-like.
- Do not reuse the same description template across users. Avoid repeating life stories such as divorce, huge debts, or lost job across many profiles.
- Avoid generating multiple users with the same life story, job loss, divorce, debt pattern, or financial situation.

Output ONLY {n_profiles} lines, one per profile, strictly using the pipe " | " delimiter, no headers, no extra commentary.
"""

# ---------------------------------------------------------------------------

generate_Twitter_monologue_prompt_ = """
You are an expert Twitter-style tweet (monologue) generator.
Generate a realistic raw Twitter tweet (monologue) about the topic: "{subtopic}"
The topic is hidden context only. Do not copy, quote, or closely paraphrase the topic wording. Express the underlying situation in your own words.
Begin immediately with a specific action, observation, consequence, setting, physical sensation, or internal thought.
Avoid generic openings such as "Felt like", "Feels like", "Just", or "Another".
The opening should immediately place the reader inside a concrete moment, action, observation, or emotional reaction.
Use the following user profile:
{profiles_block}

## STYLE EXAMPLES (Use the examples only to estimate informality, brevity, and emotional realism.
Do not reuse their wording, opening structure, first verb, user ID,or any sequence of 3 or more words.):
## EXAMPLE tweet (monologue) 1:
{sample_monologue_1}

## EXAMPLE tweet (monologue) 2:
{sample_monologue_2}  

## EXAMPLE tweet (monologue) 3:
{sample_monologue_3}

## INSTRUCTIONS to be strictly followed when generating tweet (monologue):
- Generate the full tweet (monologue) ONLY between <monologue> and </monologue> tags ONLY. No explanations. No extra text. No code fences.
- Use only the user ID listed in ({USER_ID}). Do not invent new user or reuse ID from examples.
- Use the user' profile to shape their voice, tendencies, stances, and behaviors (implicitly), but NEVER quote or reveal it verbatim.
- The user tweet:
  - must start by user ID (e.g. U0001: ),
  - must be emotionally expressive and between {OPENER_MIN_WORDS} to {OPENER_MAX_WORDS} words.
- Topic anchoring: Ground the monologue in the underlying situation through natural actions, thoughts, or emotions rather than repeating topic keywords.
- The monologue must revolve around the topic implicitly (tone, mood, behavior, conflict).
- Choose a natural tone appropriate for the user's profile and situation. Possible tones include supportive, dismissive, sarcastic, confused, bitter, numb, regretful, or angry.
- Emoji density: Include {EMOJI_MIN} to {EMOJI_MAX} emojis spread naturally across the post. Avoid clustering all at the end.
- Reflect behaviors like avoidance, rationalizing, denial, mood swings, secrecy, or tension.
- The user's behaviour should emerge naturally through decisions, reactions, priorities, and reasoning rather than explicit emotional statements alone.
- Use social-media tone (slang, emojis, abbreviations, memes). Avoid formal advice or therapy language.
- Keep most sentences short and minimal; avoid links or hashtags.
- User should reflect their risk level (e.g., high-risk may deflect/justify; low-risk may warn/reflect).

## RISK LEVEL — EXPRESSION RULES
Use the user's risk level implicitly. Never write "risk level", "high risk", "low risk", or similar labels.
NONE:
- The user should sound casual, observational, controlled, or reflective.
- Avoid chasing, secrecy, borrowing money, strong loss-of-control, or repeated-play cues.
- Losses, if mentioned, should be accepted or treated lightly.

LOW:
- The user may show temptation, mild frustration, or small slips.
- Still show some control, limits, humour, or self-awareness.
- Avoid severe consequences.

MODERATE:
- The user may rationalize, chase small losses, hide feelings, or struggle with limits.
- Emotional and money tension can appear, but avoid extreme collapse.

HIGH:
- The user may justify, deflect, chase losses, hide behaviour, raise limits, or feel pulled back.
- Emotional urgency, repeated play, secrecy, or financial pressure may appear.
- Avoid clinical labels and extreme life-collapse wording.

- Even when the topic name is explicit (e.g., "Gambling Addiction signs"), do NOT use diagnostic labels; express the idea indirectly via behaviors/emotions.

## CONDITIONING PRIORITY
When space is limited, prioritize:
1. User risk level
2. UPPS-P personality traits
3. Topic
4. Emotions and tone
5. Reflection cue
6. Money or brand details
7. Behaviour cues

- The user's personality and risk level must shape why the user reacts this way, not only what words they use.
- If the topic, risk level, personality, and behavioural cues conflict, always preserve psychological consistency over stylistic variety.
- Behaviour cues should be used only when they are consistent with the user's risk level and profile.
- When generating this single-user monologue, let the user's risk level and personality traits shape their wording, emotional intensity, justification style, and behavioural cues.

## PERSONALITY (UPPS-P) — EXPRESSION RULES
For the user above, you are given UPPS-P trait labels (high/mid/low).
Use the traits implicitly. Never mention UPPS-P, trait names, or scores.

Negative Urgency:
- HIGH → show impulsive reaction under stress, anger, sadness, shame, frustration, or anxiety.
- MID → mild emotional impulsivity is allowed.
- LOW → show more control when upset; avoid dramatic loss-of-control wording.

Positive Urgency:
- HIGH → show impulsive action after excitement, winning, confidence, hype, or celebration.
- MID → occasional excitement-driven decision is allowed.
- LOW → avoid overexcited impulsive reactions.

Lack of Premeditation:
- HIGH → show quick action, little planning, weak consideration of consequences.
- MID → mixed planning; some hesitation is allowed.
- LOW → show thinking before acting, caution, limits, or checking consequences.

Lack of Perseverance:
- HIGH → show difficulty sticking to limits, promises, plans, or stopping points.
- MID → inconsistent follow-through is allowed.
- LOW → show better follow-through or ability to stick with intentions.

Sensation Seeking:
- HIGH → show attraction to thrill, risk, novelty, fast games, big moments, or intense excitement.
- MID → moderate interest in excitement.
- LOW → avoid thrill-seeking language; prefer routine, calm, or low-risk choices.

Expression rule:
- The user's UPPS-P traits should influence how they think, decide, react emotionally, justify behaviour, and interpret events. The traits should influence decisions, reasoning, priorities, and emotional reactions rather than simply adding emotional vocabulary.
- If no trait is HIGH, reflect the general low/mid pattern subtly.
- Do not force all five traits into one short tweet.

Overrides:
- If lack_of_premeditation is LOW, avoid impulsive "didn't think" phrasing.
- If lack_of_perseverance is LOW, avoid failing-to-stick-with-it phrasing.
- If sensation_seeking is LOW, avoid thrill/novelty/risk-seeking phrasing.

## HARD LANGUAGE RULES
- Avoid therapy/clinical/diagnostic wording and self-labels. Forbidden: addiction, addicted, problem gambling, diagnosis, disorder,"practice mindfulness", "seek help",
  "I can't stop", "I need help", "I have a gambling problem", "ruined my life", "lost everything", "recovery (journey)", "hit rock bottom".
- Prefer indirect expressions like: "not great for me", "keeps pulling me back", "trying to change", "rough patch".
- Do not repeat generic sentences.
- If you feel an urge to write a clinical statement, paraphrase to indirect:
  e.g., "ruined my life" →"not great for me", "I can't stop" → "keeps pulling me back".
"""



generate_Twitter_monologue_prompt = """
You are an expert Twitter-style tweet (monologue) generator.
Generate a realistic raw Twitter tweet (monologue) about the topic: "{subtopic}"
## TOPIC SEPARATION — STRICT
- Do not quote or lightly rewrite the hidden situation.
- Do not use any sequence of 3 or more consecutive words from it.
- Do not begin with its first word or first two words.
- Express the situation through a different concrete moment, consequence,  decision, setting, observation, or reaction.
- The result must sound like an independent post, not a rewritten topic label.

The first sentence must begin with one of the following:
- a concrete action,
- a physical sensation,
- a specific observation,
- a consequence,
- a fragment of dialogue,
- a decision already in progress.

Vary the opening naturally across posts. Do not repeatedly use the same opening verb or sentence pattern.

Use the following user profile:
{profiles_block}

## STYLE EXAMPLES (for tone/structure ONLY — ignore example user ID, Do not copy their wording):
## EXAMPLE tweet (monologue) 1:
{sample_monologue_1}

## EXAMPLE tweet (monologue) 2:
{sample_monologue_2}  

## EXAMPLE tweet (monologue) 3:
{sample_monologue_3}

## INSTRUCTIONS to be strictly followed when generating tweet (monologue):
- Generate the full tweet (monologue) ONLY between <monologue> and </monologue> tags ONLY. No explanations. No extra text. No code fences.
- Use only the user ID listed in ({USER_ID}). Do not invent new user or reuse ID from examples.
- Use the user' profile to shape their voice, tendencies, stances, and behaviors (implicitly), but NEVER quote or reveal it verbatim.
- The user tweet:
  - must start by user ID (e.g. U0001: ),
  - must be emotionally expressive and between {OPENER_MIN_WORDS} to {OPENER_MAX_WORDS} words.
- Topic anchoring: Ground the monologue in the underlying situation through natural actions, thoughts, or emotions rather than repeating topic keywords.
- The monologue must revolve around the topic implicitly (tone, mood, behavior, conflict). Avoid naming the issue directly unless subtle.
- Choose a natural tone appropriate for the user's profile and situation. Possible tones include supportive, dismissive, sarcastic, confused, bitter, numb, regretful, or angry.
- Emoji density: Include {EMOJI_MIN} to {EMOJI_MAX} emojis spread naturally across the post. Avoid clustering all at the end.
- Use avoidance, rationalizing, denial, secrecy, or tension only when they are consistent with the user's risk level and personality.
- The user's behaviour should emerge naturally through decisions, reactions, priorities, and reasoning rather than explicit emotional statements alone.
- Use social-media tone (slang, emojis, abbreviations, memes). Avoid formal advice or therapy language.
- Keep most sentences short and minimal; avoid links or hashtags.
- Even when the topic name is explicit (e.g., "Gambling Addiction signs"), do NOT use diagnostic labels; express the idea indirectly via behaviors/emotions.

## RISK LEVEL — EXPRESSION RULES
Use the user's risk level implicitly. Never write "risk level", "high risk", "low risk", or similar labels.
Risk level = NONE:
- The user should sound casual, observational, controlled, or reflective.
- Avoid chasing, secrecy, borrowing money, strong loss-of-control, or repeated-play cues.
- Losses, if mentioned, should be accepted or treated lightly.

Risk level = LOW:
- The user may show temptation, mild frustration, or small slips.
- Still show some control, limits, humour, or self-awareness.
- Avoid severe consequences.

Risk level = MODERATE:
- The user may rationalize, chase small losses, hide feelings, or struggle with limits.
- Emotional and money tension can appear, but avoid extreme collapse.

Risk level = HIGH:
- The user may justify, deflect, chase losses, hide behaviour, raise limits, or feel pulled back.
- Emotional urgency, repeated play, secrecy, or financial pressure may appear.
- Avoid clinical labels and extreme life-collapse wording.

## PERSONALITY (UPPS-P) — EXPRESSION RULES
You are given UPPS-P trait labels (high/mid/low) for the user above. Use the traits implicitly. Never mention UPPS-P, trait names, or scores.

Negative Urgency:
- HIGH → show impulsive reaction under stress, anger, sadness, shame, frustration, or anxiety.
- MID → mild emotional impulsivity is allowed.
- LOW → show more control when upset; avoid dramatic loss-of-control wording.

Positive Urgency:
- HIGH → show impulsive action after excitement, winning, confidence, hype, or celebration.
- MID → occasional excitement-driven decision is allowed.
- LOW → avoid overexcited impulsive reactions.

Lack of Premeditation:
- HIGH → show quick action, little planning, weak consideration of consequences.
- MID → mixed planning; some hesitation is allowed.
- LOW → show thinking before acting, caution, limits, or checking consequences.

Lack of Perseverance:
- HIGH → show difficulty sticking to limits, promises, plans, or stopping points.
- MID → inconsistent follow-through is allowed.
- LOW → show better follow-through or ability to stick with intentions.

Sensation Seeking:
- HIGH → show attraction to thrill, risk, novelty, fast games, big moments, or intense excitement.
- MID → moderate interest in excitement.
- LOW → avoid thrill-seeking language; prefer routine, calm, or low-risk choices.

Expression rule:
- The user's UPPS-P traits should influence how they think, decide, react emotionally, justify behaviour, and interpret events. The traits should influence decisions, reasoning, priorities, and emotional reactions rather than simply adding emotional vocabulary.
- If no trait is HIGH, reflect the general low/mid pattern subtly.
- Do not force all five traits into one short tweet.

Overrides:
- If lack_of_premeditation is LOW, avoid impulsive "didn't think" phrasing.
- If lack_of_perseverance is LOW, avoid failing-to-stick-with-it phrasing.
- If sensation_seeking is LOW, avoid thrill/novelty/risk-seeking phrasing.

## Priority (highest to lowest)::
1. User risk level
2. UPPS-P personality traits
3. Topic (Hidden situation)
4. Emotions and tone
5. Reflection cue
6. Money or brand details
7. Behaviour cues

- The user's personality and risk level must shape why the user reacts this way, not only what words they use.
- If the topic, risk level, personality, and behavioural cues conflict, always preserve psychological consistency over stylistic variety.
- Behaviour cues should be used only when they are consistent with the user's risk level and profile.
- When generating the monologue, let the user's risk level and personality traits shape their wording, emotional intensity, justification style, and behavioural cues.


## HARD LANGUAGE RULES
- Avoid therapy/clinical/diagnostic wording and self-labels. Forbidden: addiction, addicted, problem gambling, diagnosis, disorder,"practice mindfulness", "seek help",
  "I can't stop", "I need help", "I have a gambling problem", "ruined my life", "lost everything", "recovery (journey)", "hit rock bottom".
- Prefer indirect expressions like: "not great for me", "keeps pulling me back", "trying to change", "rough patch".
- Do not repeat generic sentences.
- If you feel an urge to write a clinical statement, paraphrase to indirect:
  e.g., "ruined my life" →"not great for me", "I can't stop" → "keeps pulling me back".
  
"""

# ---------------------------------------------------------------------------

engagingness_system_prompt_M = """
You will be given a Twitter-style monologue (post). 
Your task is to rate the monologue on one metric.
Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation criteria:
Engagingness (1-3) Is the monologue dull/interesting?
- A score of 1 (dull) means that the monologue is generic and dull.
- A score of 2 (somewhat interesting) means the monologue is somewhat interesting and could engage you in the monologue (e.g., an opinion, thought)
- A score of 3 (interesting) means the monologue is very interesting or presents an interesting fact
Evaluation Steps:
1. Read the monologue carefully.
2. Rate the monologue on a scale of 1-3 for engagingness, according to the criteria above.

Response should just be the score like in the following. Strictly nothing else
{score}
"""

# ---------------------------------------------------------------------------

naturalness_system_prompt_M = """
You will be given a Twitter-style monologue. 
Your task is to rate the monologue on one metric.
Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation criteria:
Naturalness (1-3). Does the monologue sound like natural, human-like language?
- A score of 1 (unnatural) means the monologue sounds robotic or awkward.
- A score of 2 (somewhat natural) means the monologue sounds mostly natural but has some awkward phrases.
- A score of 3 (natural) means the monologue sounds like it has been written by a real person, with natural language flow.
Evaluation Steps:
1. Read the monologue carefully.
2. Rate the monologue on a scale of 1-3 for naturalness, according to the criteria above.

Response should just be the score like in the following. Strictly nothing else
{score}
"""

# ---------------------------------------------------------------------------

coherence_system_prompt_M = """
You will be given a Twitter-style monologue. 
Your task is to rate the monologue on one metric.
Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation criteria:
Coherence (1-3) Is the monologue logically structured and easy to follow?
- A score of 1 (incoherent) means the monologue is confusing and hard to follow.
- A score of 2 (somewhat coherent) means the monologue makes sense but has minor logical inconsistencies or unclear parts.
- A score of 3 (coherent) means the monologue is logically consistent and easy to follow.
Evaluation Steps:
1. Read the monologue carefully.
2. Rate the monologue on a scale of 1-3 for coherence, according to the criteria above.

Response should just be the score like in the following. Strictly nothing else
{score}
"""

# ---------------------------------------------------------------------------

groundedness_system_prompt_M = """
You will be given a Twitter-style monologue. 
Your task is to rate the monologue on one metric.
Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation criteria:
Groundedness (1-3) Is the monologue realistic and factually plausible?
- A score of 1 (ungrounded) means the monologue contains false or misleading information.
- A score of 2 (somewhat grounded) means the monologue is mostly accurate but may have minor errors or unsupported claims.
- A score of 3 (grounded) means the monologue is factually accurate and well-supported by evidence.
Evaluation Steps:
1. Read the monologue carefully.
2. Rate the monologue on a scale of 1-3 for groundedness, according to the criteria above.

Response should just be the score like in the following. Strictly nothing else
{score}
"""

# ---------------------------------------------------------------------------

judge_prompt_M = """
You are an expert evaluator of Twitter-style personal posts.
You will be given a self-written Twitter post (monologue) by a user along with their post topic.
Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Your task is to evaluate the quality of the monologue based on the following dimensions:

## Dimensions
Coverage of the topic - Does the post reflect the theme and explore meaningful aspects of the topic?
Groundedness - Does the post feel honest, realistic, and grounded in real-life experience?
Naturalness - Does it read like something a real person might write on Twitter?
Emotional impact - Does it express emotion (e.g., shame, regret, frustration) in a relatable way?
Coherence - Is the post clear and logically structured?
Self-reflection - Does it show any internal struggle, thought process, or personal insight?

Based on these dimensions, grade the post with a score between 1 and 10.

Follow the template given below for your response.
###### Response Template Starts ######
Groundedness - <explanation about whether the post is grounded/not grounded>
Naturalness - <explanation on how natural or artificial the post feels>
Emotional impact - <discussion on the emotional expression or depth>
Coherence - <notes on clarity and logical flow>
Self-reflection - <comments on insight or introspection>
Score: <integer score between 1 to 10>
##### Response Template Ends #####

Each dimension should be separated by a new line.
The last line of the response should be your grade.
"""