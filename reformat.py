lines = open("input").readlines()
with open("results.txt", "w") as f:
    for line in lines:
        line = line.strip()
        if line.startswith("("):
            # ('1hzz', 0.8457), ('1pjb', 0.8735) format is like this
            pred1 = line.split(",")[0][1:].replace("'", "")
            score1 = float(line.split(",")[1].split(")")[0])
            pred2 = line.split(",")[2][2:].replace("'", "")
            score2 = float(line.split(",")[3].split(")")[0])
            f.write("\t".join([pred1, pred2, str(score1), str(score2)]) + "\n")