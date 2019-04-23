import matplotlib.pyplot as plt
import os

data_dir = "C:\\Users\\chkar\\Desktop\\speech_project_saves\\accuracies"
colours = ["b-", "g-", "r-", "y-", "m-", "k-", "c-"]



for i in range(4):
    accuracies = []
    training_steps = []
    with open(os.path.join(data_dir, "acc_%d" %i )) as fp:  
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