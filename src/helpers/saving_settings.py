



DIR = "src\helpers\\txt_files\settings.txt"



def save_settings(settings_tag, settings_value, directory = DIR):
    with open(directory, "r") as file:
        for i, line in enumerate(file):
            if settings_tag in line:
                line_index = i
    
    with open(directory, "w") as file:
        for i, line in enumerate(file):
            if i == line_index:
                file.write(f"{settings_tag}:{settings_value}\n")

def load_settings(settings_tags, directory = DIR):
    settings_values = [0]*len(settings_tags)
    with open(directory, "r") as file:
        for line in file:
            for i, tag in enumerate(settings_tags):
                if tag in line:
                    value = line.split("=")[1].strip()
                    settings_values[i] = value
    return settings_values