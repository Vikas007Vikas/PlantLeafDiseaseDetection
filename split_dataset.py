import splitfolders

splitfolders.ratio("./segmented_saliency_imposed", # The location of dataset
                   output="segmented_data_80_20", # The output location
                   seed=42, # The number of seed
                   ratio=(.8, .2), # The ratio of splited dataset
                   group_prefix=None, # If your dataset contains more than one file like ".jpg", ".pdf", etc
                   move=False # If you choose to move, turn this into True
                   )
