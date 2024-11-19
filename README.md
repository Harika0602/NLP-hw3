This repository evaluates the ranking abilities of OpenAI's GPT models (e.g., GPT-3.5-turbo) on sequential recommendation datasets like ml-1m and lastfm.

1.	Unzip dataset files.
2.	cd LLMSRec_Syn/dataset/ml-1m/; unzip ml-1m.inter.zip
    
    For data preparation details, please refer to LLMRank's [data-preparation](https://github.com/RUCAIBox/LLMRank/blob/master/llmrank/dataset/data-preparation.md).
3.  Install dependencies.
    ```bash
    pip install -r requirements.txt
4.	Insert your OpenAI key on line 20 of ./model/rank.py.
5.	Evaluate OpenAI GPT (GPT-3.5-turbo)'s ranking abilities on ml-1m / (lastfm) dataset.
   
    cd LLMSRec_Syn/

    To run the evaluation without any noise injection 
    ```bash
    python run_test.py -m RankAggregated -d ml-1m / (lastfm)

    python run_test.py -m RankNearest -d ml-1m / (lastfm)

    python run_test.py -m RankFixed -d ml-1m / (lastfm)

6.	To run the evaluation with noise injection 
    ```bash
    python run_test.py -m RankAggregated -d ml-1m --noise_type random --noise_ratio 0.3

    python run_test.py -m RankAggregated -d ml-1m --noise_type truncate --noise_ratio 0.3

    python run_test.py -m RankAggregated -d ml-1m --noise_type Duplicate --noise_ratio 0.3
7.	We also tried for different seeds.
    ```bash
    python run_test.py -m RankNearest -d ml-1m --noise_type duplicate --noise_ratio 0.3 -sd 42

    python run_test.py -m RankAggregated -d ml-1m / (lastfm) -sd 123
     
