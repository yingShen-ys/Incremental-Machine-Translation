# Weekly Updates

This document contains the weekly updates on the incremental machine translation project for the MIIS Capstone requirement.

### Work done since the final presentation in May

<ul>
    <li>
        Continued literature review on machine translation and word reordering strategies.
    </li>
    <li>Discussed with the advisor on finding more spoken language translation datasets that require word reordering besides machine translation datasets on written text.</li>
</ul>

### Sept 5, 2018 - Sept 12, 2018

| Member  | Upcoming Tasks                                               | Completed Tasks                                        |
| ------- | ------------------------------------------------------------ | ------------------------------------------------------ |
| Ying    | Searching for spoken language translation datasets.<br />Literature review on word reordering in real-time Machine Translation. | Found five spoken language translation datasets.<br /> |
| Zhun    | Searching for spoken language translation datasets.<br />Literature review on word reordering in real-time Machine Translation. | Found five spoken language translation datasets.       |
| Kangyan | Searching for spoken language translation datasets.<br />Learning how to deploy machine learning models on AWS. | Investigated how to deploy tensorflow model.           |

### Sept 12, 2018 - Sept 19, 2018

| Member  | Upcoming Tasks                                               | Completed Tasks                                      |
| ------- | ------------------------------------------------------------ | ---------------------------------------------------- |
| Ying    | Further searching spoken language translation datasets in real-time translation setup. | Found two simultaneous interpretation datasets.      |
| Zhun    | Searching for other parallel corpora that comes from the source of speech involving non-SVO languages. | Found two simultaneous interpretation datasets.      |
| Kangyan | Still working on how to deploy machine learning models on AWS. | Finish investigating how to deploy tensorflow model. |

### Sept 19, 2018 - Sept 26, 2018

| Member  | Upcoming Tasks                                               | Completed Tasks                                              |
| ------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Ying    | Doing a literature review on the topic of incremental translation.<br />Investigating the human strategies in simultaneous interpretation. | Located two interpretation dataset.<br />Investigated human strategies in simultaneous interpretation.<br />Get accessed to three dataset that requires agreement. |
| Zhun    | Doing a literature review on the topic of incremental translation. | Located two interpretation dataset.<br />Put up a reading list on incremental translation for the team. |
| Kangyan | Start working on backend from Xinjian's existing code.       | Worked on backend from Xinjian's existing code.              |

### Sept 26, 2018 - Oct 3, 2018

| Member  | Upcoming Tasks                                               | Completed Tasks                                              |
| ------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Ying    | Investigating evaluation metrics for simutaneous interpretation.<br />Writing a summary of possible datasets. | Completed the summary for possible datasets.<br />Decided to go with the NAIST Japanese-English simutaneous interpretation datasets.<br />Implemented the basic machine translation model. |
| Zhun    | Getting access to some of the datasets that need consent forms, signing agreements, etc. | Updated a list of related works that we can base our work on.<br />Implemented the basic machine translation model. |
| Kangyan | Working on backend from Xinjian's existing code.             | Implemented the basic machine translation model.             |

### Oct 3, 2018 - Oct 10, 2018

| Member  | Upcoming Tasks                                               | Completed Tasks                                              |
| ------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Ying    | Train and test the basic machine translation model on the NAIST dataset.<br />Keep requesting the CIAIR corpus from the authors. | Trained and tested the basic machine translation model on the JESC dataset. |
| Zhun    | Implementation of a baseline neural machine translation model | Implemented the baseline neural machine translation model    |
| Kangyan | Working on backend from Xinjian's existing code.             | Worked on backend from Xinjian's existing code.              |

### Oct 10, 2018 - Oct 17, 2018

| Member  | Upcoming Tasks                                               | Completed Tasks                                           |
| ------- | ------------------------------------------------------------ | --------------------------------------------------------- |
| Ying    | Implement the baseline incremental neural machine translation model. | Implemented the baseline neural machine translation model |
| Zhun    | Implement the baseline incremental neural machine translation model. | Implemented the baseline neural machine translation model |
| Kangyan | Working on backend from Xinjian's existing code.             | Worked on backend from Xinjian's existing code.           |

### Oct 10, 2018 - Oct 17, 2018

| Member  | Upcoming Tasks                                               | Completed Tasks                                           |
| ------- | ------------------------------------------------------------ | --------------------------------------------------------- |
| Ying    | Implement the baseline incremental neural machine translation model. | Implemented the baseline neural machine translation model |
| Zhun    | Pretrain non-incremental Ja-En MT system | Finished pretraining of the model |
| Kangyan | Working on backend from Xinjian's existing code.             | Worked on backend from Xinjian's existing code.           |

### Oct 17, 2018 - Oct 24, 2018

| Member  | Upcoming Tasks                                   | Completed Tasks                                              |
| ------- | ------------------------------------------------ | ------------------------------------------------------------ |
| Ying    | Examing the simultaneous interpretation data.    | Discovered that interpretation datasets are not readily aligned for guiding model as expert |
| Zhun    | Prerocess incremental interpretation datasets    | Discovered that interpretation datasets are not readily aligned for guiding model as expert |
| Kangyan | Working on backend from Xinjian's existing code. | Worked on backend from Xinjian's existing code.              |

### Oct 24, 2018 - Oct 31, 2018

| Member  | Upcoming Tasks                                               | Completed Tasks                                              |
| ------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Ying    | Exploring data augmentation methods for enlarging the current corpus. | Explored data augmentation methods.                          |
| Zhun    | Running GIZA++ for word alignments on non-incremental datasets | Run GIZA++ for word alignments on non-incremental datasets   |
| Kangyan | Working on backend from Xinjian's existing code.             | Training the baseline incremental neural machine translation model. |

### Oct 31, 2018 - Nov 7, 2018

| Member  | Upcoming Tasks                                               | Completed Tasks                                              |
| ------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Ying    | Running fast alignment on the current corpus and exploring the back translation model to enlarge the corpus. | Ran fast alignment on the simultaneous interpretation corpus and installed the statistical machine translation model for back translation. |
| Zhun    | Deriving expert action sequence for RL agent from aligned bilingual corpus | Re-ran alignments using a combined corpus of incremental and non-incremental translation datasets to obtain more reliable alignments for the interpretation dataset. |
| Kangyan | Implementing the incremental neural machine translation model. | Debugging the incremental translation model                  |

### Nov 7, 2018 - Nov 14, 2018

| Member  | Upcoming Tasks                                               | Completed Tasks                                              |
| ------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Ying    | Exploring data augmentation                                  | Run syntax-based rewriting on JA-EN corpus                   |
| Zhun    | Deriving expert action sequence for RL agent from aligned bilingual corpus | Switched to debugging reinforcement learning agent since it is not working properly |
| Kangyan | Implementing the incremental neural machine translation model. |                                                              |

### Nov 14, 2018 - Nov 21, 2018

| Member  | Upcoming Tasks                                               | Completed Tasks |
| ------- | ------------------------------------------------------------ | --------------- |
| Ying    | Implementing incremental neural machine translation mode     |                 |
| Zhun    | Debugging the reinforcement learning agent                   |                 |
| Kangyan | Implementing another incremental neural machine translation model. |                |

### Nov 21, 2018 - Nov 28, 2018

| Member  | Upcoming Tasks                                               | Completed Tasks |
| ------- | ------------------------------------------------------------ | --------------- |
| Ying    | Running incremental neural machine translation with rewritten data.     |                 |
| Zhun    | Debugging the reinforcement learning agent.                   |                 |
| Kangyan | Working on system combination with Xinjian. | Implemented another incremental neural machine translation model.               |
