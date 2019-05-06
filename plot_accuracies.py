import matplotlib.pyplot as plt
import os

data_dir = "C:\\Users\\chkar\\Desktop\\speech_project_saves\\training_session_1\\accuracies"
colours = ["b-", "g-", "r-", "y-", "m-", "k-", "c-", "brown"]
acc_files = ["acc_0", 
             "acc_1", 
             "acc_2", 
             "acc_3", 
             "acc_4", 
             "acc_5", 
             # "acc_0_lrdec_10000", 
             # "acc_1_lrdec_10000"
            ]


for i in range(len(acc_files)):
    accuracies = []
    training_steps = []
    with open(os.path.join(data_dir, acc_files[i])) as fp:  
        line = fp.readline()
        while line:
            if "index" in line:
                first_split = line.split("-")
                accuracies.append(float(first_split[0].split("_")[-1]))
                training_steps.append(int(first_split[-1].split(".")[0]))
            line = fp.readline()
            
        plt.plot(training_steps, accuracies, colours[i])
        
        
plt.ylabel('accuracy')
plt.xlabel('training step')
plt.show()