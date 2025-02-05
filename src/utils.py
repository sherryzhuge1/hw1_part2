import yaml
import wandb

PHONEMES = [
    "[SIL]",
    "AA",
    "AE",
    "AH",
    "AO",
    "AW",
    "AY",
    "B",
    "CH",
    "D",
    "DH",
    "EH",
    "ER",
    "EY",
    "F",
    "G",
    "HH",
    "IH",
    "IY",
    "JH",
    "K",
    "L",
    "M",
    "N",
    "NG",
    "OW",
    "OY",
    "P",
    "R",
    "S",
    "SH",
    "T",
    "TH",
    "UH",
    "UW",
    "V",
    "W",
    "Y",
    "Z",
    "ZH",
    "[SOS]",
    "[EOS]",
]


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_wandb(config):
    wandb.login(key="fd2f4a747b6d6934bcc41013afb25a02f4753a3e")
    return wandb.init(
        name    = "run-19", ### Wandb creates random run names if you skip this field, we recommend you give useful names
        reinit  = True, ### Allows reinitalizing runs when you re-run this cell
        # id     = "xgesgg8t", ### Insert specific run id here if you want to resume a previous run
        # resume = "must", ### You need this to resume previous runs, but comment out reinit = True when using this
        project = "hw1p2", ### Project should be created in your wandb account
        config  = config ### Wandb Config for your run
    )


def save_predictions(predictions, filename="submission.csv"):
    with open(filename, "w+") as f:
        f.write("id,label\n")
        for i, pred in enumerate(predictions):
            f.write(f"{i},{pred}\n")
