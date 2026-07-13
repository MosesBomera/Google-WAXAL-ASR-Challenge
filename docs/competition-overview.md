# Can your model advance speech AI for underrepresented languages?

Voice technologies have transformed how people interact with digital services, yet most African languages remain underserved due to a lack of large-scale, high-quality speech datasets. WAXAL changes that. Developed through a multi-year collaboration between Google Research and multiple African academic and community organisations, WAXAL provides one of the largest openly-accessible speech resources for African languages, covering 27 languages spoken by more than 100 million people across Africa. The dataset includes thousands of hours of natural speech and high-quality recordings designed to advance speech recognition and speech generation research. This challenge aims to accelerate the development of inclusive speech technologies that enable more people to access digital services in their own languages.

In this challenge, you will use the WAXAL dataset to build automatic speech recognition (ASR) systems that can generalise to previously unseen speech data, with a focus on Lingala, Shona, and Luganda.

## About Google Research

At Google Research, our mission is to drive breakthroughs that benefit society, businesses, and Google products. We advance scientific progress by publishing hundreds of papers each year and collaborating with universities, NGOs, partners, and communities worldwide. Our goal is to build a world where AI is more than a tool: it’s an essential partner for researchers, scientists, clinicians, teachers, users, and businesses.

## Evaluation

This challenge uses multi-metric evaluation. There are two error metrics: Word Error Rate (WER) and Character Error Rate (CER).

The final score on the leaderboard is the weighted mean of the two evaluation metrics.

| Metric | Weighting |
|---|---|
| WER | 0.5 |
| CER | 0.5 |

This balances word-level accuracy with character-level robustness, helping account for spelling variation across African languages.

## Prizes

- 🥇 1st prize: $4 000 USD
- 🥈 2nd prize: $2 500 USD
- 🥉 3rd prize: $2 000 USD
- Best African participant: $1 500 USD

Note: best African participant can also win an overall prize.

There are 10 000 Zindi points available. You can read more about Zindi points here.

## Rules

- Languages and tools: You may only use open-source languages and tools in building models for this challenge.
- Who can compete: Open to all
- Submission limits: 5 submissions per day, 200 submissions overall.
- Team size: Max team size of 4
- Public-private split: Zindi maintains a public leaderboard and a private leaderboard for each challenge. The public leaderboard includes approximately 30% of the test dataset. The private leaderboard will be revealed at the close of the challenge and contains the remaining 70% of the test set.
- Data sharing: CC-BY 4.0
- Code review: Top 10 on the private leaderboard will receive an email requesting their code at the close of the challenge. You will have 48 hours to submit your code.
- Code sharing: Multiple accounts, or sharing of code and information across accounts not in teams, is not allowed and will lead to disqualification.
