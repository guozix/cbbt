
def get_template(DATASET_NAME):
    if DATASET_NAME == "EuroSAT":
        eurosat_dataset_prompt = "a centered satellite photo of {}."
        eurosat_object_categories = ["AnnualCrop","Forest","HerbaceousVegetation","Highway","Industrial","Pasture","PermanentCrop","Residential","River","SeaLake"]
        return eurosat_dataset_prompt, eurosat_object_categories
