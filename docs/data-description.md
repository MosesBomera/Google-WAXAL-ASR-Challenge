# About

## Phase 1: Build, Experiment, and Climb the Leaderboard

Welcome to the challenge! In Phase 1, you'll explore the WAXAL dataset and build speech recognition models for African languages using one of the largest openly available African speech resources ever created.

Phase 1 uses the WAXAL train, validation, and test splits as provided on Hugging Face. You will have access to the training and validation data, including transcriptions, for model development. The provided test set will be used for leaderboard evaluation, with participants submitting predicted transcriptions for scoring.

This gives you the opportunity to experiment with different architectures, fine-tuning approaches, data augmentation techniques, and multilingual learning strategies. Once your model is ready, you'll generate predictions for the provided test set and submit them to the leaderboard.

This is your chance to learn from the data, compare approaches with the community, and steadily improve your score. Whether you're building your first ASR model or pushing the state of the art, Phase 1 is all about innovation, collaboration, and discovering what works.

Use this phase to develop the strongest model you can — we'll be putting it to the ultimate test in Phase 2.

Participants may supplement the provided challenge data with other publicly available open-source speech or language datasets. Any external datasets used must be:

- publicly accessible
- legally licensed for research or development
- disclosed in the final solution documentation

## Phase 2: The Ultimate Generalisation Test

The real challenge begins here. At the start of Phase 2, we'll release a completely new and unseen test set. These audio samples have not been included in any of the training, validation, or Phase 1 test data, providing a true measure of how well your model generalises to new speakers and recordings.

Your task is simple: use the model you've developed during Phase 1 to generate predictions for this new dataset. No additional labels will be provided, and the final competition rankings will be determined using performance on this unseen dataset.

To ensure a true test of model generalisation, participants will only receive the audio data during Phase 2 — approximately one week before the challenge closes. Metadata and auxiliary information such as language, speaker identity, gender, and other descriptive attributes will not be provided. Successful solutions will therefore need to rely on the speech signal itself rather than metadata-driven shortcuts.

This phase rewards robust, well-designed models rather than leaderboard optimisation. The teams that have learned the most from the WAXAL dataset and built solutions that generalise effectively across languages and speakers will rise to the top.

> Important: The Phase 1 leaderboard is designed to support model development, collaboration, and experimentation. Final rankings and prize winners will be determined based on performance on the Phase 2 evaluation dataset.

Any Phase 1 submission that uses the publicly available ground-truth labels for the Phase 1 test set will be treated as a breach of the challenge rules and may lead to disqualification.

## Data Access

Access the data here: https://huggingface.co/datasets/google/WaxalNLP
