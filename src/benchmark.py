# import os
import json
from matplotlib import pyplot as plt
import numpy as np
from scipy.stats import norm

def test_score(algorithm, rag_searcher, evaluator) -> float:
    # load dataset from dataset/question.json
    with open("dataset/question.json") as f:
        dataset = json.load(f)
    
    language = ["Chinese", "English"]
    score_list = []
    # dataset.flatten(2)
    for test_data in dataset:
        question_title = test_data["Q"][0][0]
        question_quest = test_data["Q"][0][1]

        score = {}
        for lang in language:
            content, reference = algorithm(
                rag = rag_searcher,
                question = {
                    "title": question_title,
                    "quest": question_quest
                },
                language = lang
            )

            score[lang] = {}
            score[lang]["content"] = evaluator["content"](
                benchmark = test_data["A"][0],
                text = content,
                language = lang
            )
            score[lang]["reference"] = evaluator["reference"](
                benchmark = test_data["R"][0],
                text = reference
            )

        score_list.append(score)
    
    return score_list


def parse_indicator(score_list, indicator: str):
    return [
        {
            lang: s[lang][indicator]
            for lang in s.keys()
        }
        for s in score_list
    ]

def plot_distribution(score_list, output_path="score_distribution.png"):
    chinese_scores = sorted([s['Chinese'] for s in score_list], reverse=True)
    english_scores = sorted([s['English'] for s in score_list], reverse=True)

    stats = {
        "Chinese": {
            "mean": np.mean(chinese_scores),
            "std": np.std(chinese_scores),
            "data": chinese_scores
        },
        "English": {
            "mean": np.mean(english_scores),
            "std": np.std(english_scores),
            "data": english_scores
        }
    }

    plt.figure(figsize=(12, 6))
    colors = {'Chinese': '#1f77b4', 'English': '#ff7f0e'}
    
    for idx, (lang, values) in enumerate(stats.items()):
        ax = plt.gca() if idx == 0 else plt.gca().twinx()
        x = np.linspace(min(values["data"])-5, max(values["data"])+5, 100)
        y = norm.pdf(x, values["mean"], values["std"])
        ax.plot(x, y, color=colors[lang], linestyle='--', label=f'{lang} Normal Distribution')

        n, bins, _ = ax.hist(
            values["data"], bins=10, density=True, 
            alpha=0.5, color=colors[lang],
            label=f'{lang} Actual Scores'
        )
        
        ax.text(
            0.95, 0.85 - idx*0.15, 
            f'{lang}:\nμ = {values["mean"]:.2f}\nσ = {values["std"]:.2f}',
            transform=ax.transAxes, color=colors[lang],
            ha='right', va='top', fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8)
        )

        ax.set_ylim(0, max(n)*1.2)
        if idx > 0:
            ax.spines['right'].set_color(colors[lang])
            ax.yaxis.label.set_color(colors[lang])
            ax.tick_params(axis='y', colors=colors[lang])
        else:
            ax.spines['left'].set_color(colors[lang])
            ax.yaxis.label.set_color(colors[lang])
            ax.tick_params(axis='y', colors=colors[lang])

    plt.title('Score Distribution')
    # plt.xlabel('Score Ranking')
    plt.xlim(min(english_scores)-5, max(chinese_scores)+5)
    plt.xticks(np.arange(min(english_scores)//10*10, max(chinese_scores)+10, 10))
    plt.grid(axis='x', alpha=0.3)
    
    lines, labels = [], []
    for ax in plt.gcf().axes:
        axLine, axLabel = ax.get_legend_handles_labels()
        lines.extend(axLine)
        labels.extend(axLabel)
    plt.legend(lines, labels, loc='upper left')

    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()