import matplotlib.pyplot as plt
import os

def parse_model_name(file):
    # get the model name from the file name
    model_name = file.split(".")[0]
    if "rank" in model_name:
        model_name = model_name.replace("_rank=", "(") + ")"
    return model_name

if __name__ == "__main__":
    results = {}

    # get all text files in the output directory
    for file in os.listdir("../output"):
        if file.endswith(".txt"):
            model_name = parse_model_name(file)
            results[model_name] = {}

            # read the file
            with open(f"../output/{file}", "r") as f:
                lines = f.readlines()

                for line in lines:
                    # try to match 'Model has 669,706 parameters'
                    if "Model has" in line:
                        # get the number of parameters
                        num_parameters = int(line.split(" ")[2].replace(",", ""))
                        results[model_name]["num_parameters"] = num_parameters
                    # try to match 'accuracy=[52.480000000000004, ...]'
                    elif "accuracy=" in line:
                        # get the list of accuracies
                        accuracies = eval(line.split("=")[1])
                        results[model_name]["accuracies"] = accuracies

    # print the results
    # for model_name, data in results.items():
    #     print(f"Model: {model_name}")
    #     print(f"  Number of parameters: {data['num_parameters']:,}")
    #     print(f"  Accuracies: {data['accuracies']}\n")
                        
    # plot the results
    plt.figure(figsize=(10, 5))
    colors = ["blue", "green", "red", "purple", "orange", "brown", "pink", "gray", "olive", "cyan"]
    # plot mlp baseline
    plt.plot(results["mlp"]["accuracies"], label="mlp", color="black")
    print(f"MLP:             {results['mlp']['num_parameters']:,} parameters")
    # plot lora models
    lora_names = [model_name for model_name in results if "lora" in model_name]
    lora_names.sort(key=lambda x: int(x.split("(")[1].split(")")[0])) # sort the model names by rank
    for i, model_name in enumerate(lora_names[2:]):
        rank = int(model_name.split("(")[1].split(")")[0])
        plt.plot(results[model_name]["accuracies"], label=model_name, color=colors[i])
        print(f"MLP+LoRA(rank={rank}): {results[model_name]['num_parameters']:,} parameters")
    # plot ae models
    ae_names = [model_name for model_name in results if "ae" in model_name]
    ae_names.sort(key=lambda x: int(x.split("(")[1].split(")")[0])) # sort the model names by rank   
    for i, model_name in enumerate(ae_names[2:]):
        # make a dashed line plot
        plt.plot(results[model_name]["accuracies"], label=model_name, color=colors[i], linestyle='dashed')
        print(f"MLP+AE(rank={rank}):   {results[model_name]['num_parameters']:,} parameters")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy")
    plt.legend()
    plt.show()