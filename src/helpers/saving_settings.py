DIR = "src\\helpers\\txt_files\settings.txt"

def save_settings(setting_tags, settings_values, directory):
    stats_file = open(directory, "r")
    stats_content = []
    all_content = []
    
    for line in stats_file:
        #remove non-number characters, then append the remaining number to stats_content
        result = ''.join([char for char in line if char.isdigit()])
        stats_content.append(result)
        all_content.append(line)
    stats_file.close()
    #updates values in file
    for i, tag in enumerate(setting_tags):
        for j, line in enumerate(all_content):
            if tag in line:
                all_content[j] = f"{tag} = {settings_values[i]}\n" 

    with open(directory, "w") as file:
        for line in all_content:
            file.write(line)

def load_settings(settings_tags, directory = DIR):
    #returns setting values of relevant lines in file
    settings_values = [0]*len(settings_tags)
    with open(directory, "r") as file:
        for line in file:
            for i, tag in enumerate(settings_tags):
                if tag in line:
                    value = line.split("=")[1].strip()
                    settings_values[i] = value
    return settings_values