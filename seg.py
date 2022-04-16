import glob
import os
import jieba
import snownlp
import thulac
from tqdm import tqdm
thu = thulac.thulac(seg_only=True)

jieba.initialize()

file_list = glob.glob("testing/*.utf8")

# perl scripts/score gold/pku_training_words.utf8 gold/pku_test_gold.utf8 results/pku_test_jieba.utf8 > score_jieba.utf8

def gen_score(file_name, res_path, method):
    final_path = "scores/score_dataset_method.utf8"
    if "pku" in file_name:
        os.system("perl {} {} {} {} > {}".format("scripts/score", "gold/pku_training_words.utf8", "gold/pku_test_gold.utf8", res_path, final_path.replace("dataset", "pku").replace("method", method)))
    elif "as" in file_name:
        os.system("perl {} {} {} {} > {}".format("scripts/score", "gold/as_training_words.utf8", "gold/as_testing_gold.utf8", res_path, final_path.replace("dataset", "as").replace("method", method)))
    elif "cityu" in file_name:
        os.system("perl {} {} {} {} > {}".format("scripts/score", "gold/cityu_training_words.utf8", "gold/cityu_test_gold.utf8", res_path, final_path.replace("dataset", "cityu").replace("method", method)))
    elif "msr" in file_name:
        os.system("perl {} {} {} {} > {}".format("scripts/score", "gold/msr_training_words.utf8", "gold/msr_test_gold.utf8", res_path, final_path.replace("dataset", "msr").replace("method", method)))

for file_name in file_list:
    with open(file_name, "r") as fr:
        datas = fr.readlines()

    # jieba 结果
    jieba_res_path = file_name.replace("testing", "results").replace(".utf8", "_jieba.utf8")
    with open(jieba_res_path, "w") as fw:
        for data in tqdm(datas):
            data = data.strip()
            if data == "":
                fw.write("\n")
                continue
            wordlist = list(jieba.cut(data))
            res = "  ".join(wordlist)
            fw.write(res+"\n")
    gen_score(file_name, jieba_res_path, "jieba")

    # snownlp 结果
    snownlp_res_path = file_name.replace("testing", "results").replace(".utf8", "_snownlp.utf8")
    with open(snownlp_res_path, "w") as fw:
        for data in tqdm(datas):
            data = data.strip()
            if data == "":
                fw.write("\n")
                continue
            wordlist = snownlp.SnowNLP(data).words
            res = "  ".join(wordlist)
            fw.write(res+"\n")
    gen_score(file_name, snownlp_res_path, "snownlp")

    # thulac 结果
    thulac_res_path = file_name.replace("testing", "results").replace(".utf8", "_thulac.utf8")
    with open(thulac_res_path, "w") as fw:
        for data in tqdm(datas):
            data = data.strip()
            if data == "":
                fw.write("\n")
                continue
            wordlist = thu.cut(data, text=True).split()
            res = "  ".join(wordlist)
            fw.write(res+"\n")
    gen_score(file_name, thulac_res_path, "thulac")
