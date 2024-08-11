import neptune
import yaml

config = yaml.safe_load(open("config.yaml"))


def delete():
    # Initialize Neptune client
    run = neptune.init_run(
        project="towards-hi/image-classification",
        api_token=config["neptune_api_token"],
        with_id="IM-254",
    )

    # Define the range of epochs to delete
    start_epoch = 105
    end_epoch = 330
    step = 5

    # Delete checkpoint files
    for epoch in range(start_epoch, end_epoch + 1, step):
        file_path = f"checkpoints/epoch_{epoch}"
        try:
            # run.delete_files(file_path)
            del run[file_path]
            print(f"Deleted {file_path}")
        # except neptune.exceptions.NeptuneException as e:
        except Exception as e:
            print(f"Error deleting {file_path}: {str(e)}")

    # Stop the run
    run.stop()


if __name__ == "__main__":
    inp = input("are you sure you want to delete the checkpoints? Y/N")
    if inp == "Y":
        delete()
    else:
        print("Exiting without deleting checkpoints.")
