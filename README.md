# GP Clincial Domain LLM Probing

### Overview

This repo contains all of the code that is needed for our GP project. 

This repository contains all necessary code and configurations for conducting probing tasks related to group project. Specifically, includes code for the BioLAMA and MedLAMA probing tasks, which are designed to evaluate the medical knowledge embedded in language models. These tasks are have been written so they work with the `lm-evaluation-harness` provided by EleutherAI. We have also designed our own MCQ versions of BioLAMA and MedLAMA to experiment with how altering the type of question affects the performance.

### Structure

The repo is structured as: 
```
.
├── BioLAMA_CTD
│   ├── all tasks associated with CTD
├── BioLAMA_UMLS
│   │   ├── all tasks associated with UMLS
├── BioLAMA_Wikidata
│   ├── all tasks associated
├── MedLAMA
│   ├── all tasks associated with MedLAMA
└── README.md
```

#### Running

If you want to run the BioLAMA and MedLAMA tasks in this repo with [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness) simply follow their instructions for getting setup. After you have cloned our repo to, you're ready to go! Make sure you include the `--include_path` pointing to the location of the task you want to run when running the experiments. We also reccomend using `--output_path` and `--log_samples` so you get lots of data to analyse after the experiment is run!
