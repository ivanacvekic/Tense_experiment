# Processing online L2 English tense/aspect agreement violations

The current project deals with online processing of present perfect and past simple tense and temporal adverbials during 
self-paced reading (SPR). The experiment aims at looking if L2 learners use the L2 English tense information incrementally and
if the existence of grammaticalized tense and aspect aids them in the process. The participants in the study are L1 German, L1 
Croatian and L1 Spanish learners of L2 English that either instantiate aspect (Croatian, Spanish) or not (German).

In the tense_participant.py script, the descriptive analysis of the participants is calculated and between-group differences 
are tested:

- Age
- Age of Acquisition (AoA)
- Years of learning English
- Test: English proficiency - LexTALE
- Test: English proficiency - Category Fluency Task (CFT)
- Test: knowledge of English tenses (Cloze test)
- Test: offline knowledge of verb tense and temporal adverbial agreement - Acceptability Judgement Task (AJT)
- Test: online knowledge of verb tense and temporal adverbial agreement - Self Paced Reading (SPR)

In the tense_main_experiment.py script, the data is cleaned from outliers, it is visualized, and within-group and between-group
differences are tested using the lmer model.
