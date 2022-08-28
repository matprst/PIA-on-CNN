import torch
import pandas as pd
import datasets
import utils
import os

def evaluate_all_models(input_file, models_dir, results_file, device):
    attribute = 'Mouth_Slightly_Open'
    train_set = datasets.get_dataset(attribute, split='train')
    train_loader = datasets.get_dataloader(train_set, property=None, size=None, class_proportions=None, batch_size=64)
    test_set = datasets.get_dataset(attribute, split='test')
    test_loader = datasets.get_dataloader(test_set, property=None, size=None, class_proportions=None, batch_size=64)

    df_models = pd.read_csv(input_file, index_col=0)
    print(df_models)
    
    data = {'model': [], 'architecture': [], 'accuracy_train': [], 'precision_train': [], 'recall_train': [], 'accuracy_test': [], 'precision_test': [], 'recall_test': []}
    architecture_ids = (f'a{i}' for i in range(1, 10))

    for architecture in architecture_ids:
        df_models_architecture = df_models[df_models['architecture'] == architecture]['model'].values
        
        if len(df_models_architecture) == 0:
            continue

        for filename in df_models_architecture:
            path = os.path.join(models_dir, filename)
            model = utils.get_model(architecture)
            model.load_state_dict(torch.load(path, map_location=torch.device(device)))

            # Evaluate on training set
            a_train, p_train, r_train = utils.evaluate(train_loader, model.to(device), device)
            data['precision_train'].append(f'{p_train:.3f}')
            data['accuracy_train'].append(f'{a_train:.3f}')
            data['recall_train'].append(f'{r_train:.3f}')

            # Evaluate on test set
            a_test, p_test, r_test = utils.evaluate(test_loader, model.to(device), device)
            data['precision_test'].append(f'{p_test:.3f}')
            data['accuracy_test'].append(f'{a_test:.3f}')
            data['recall_test'].append(f'{r_test:.3f}')

            data['model'].append(filename)
            data['architecture'].append(architecture)

        # Save results
        new_df = pd.DataFrame(data)
        print(new_df)
        new_df.to_csv(results_file, mode='a', header=True)
        data = {'model': [], 'architecture': [], 'accuracy_train': [], 'precision_train': [], 'recall_train': [], 'accuracy_test': [], 'precision_test': [], 'recall_test': []}


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--shadow_csv", type=str, default="./template_output.csv", help="path to csv containing the info about the shadow models.")
    parser.add_argument("--results_csv", type=str, default="./models_performance.csv", help="csv file where the performance of the shadow models will be output.")
    parser.add_argument("--models_dir", type=str, default="./models/shadow_models", help="path to the directory that contains the states of the shadow models.")
    parser.add_argument("--cuda", action="store_true", default=False, help="use GPU.")

    args = parser.parse_args()


    device = torch.device("cuda:0" if args.cuda else "cpu")

    attribute = 'Mouth_Slightly_Open'
    hidden_attribute = 'Male'

    evaluate_all_models(
        input_file=args.shadow_csv,
        models_dir=args.models_dir,
        results_file=args.results_csv,
        device=device
    )


if __name__ == "__main__":
    main()

