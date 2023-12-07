import typing
import data
from pipeline import Pipeline
import pipeline


class IsolatedPipline(Pipeline):
    def run(self, *, 
            train_documents: typing.List[data.Document],
            test_documents: typing.List[data.Document], training_only: bool = True): # Adding a flag to only train the model
        
        print(f'Running {self.description()}')
        train_documents = [d.copy() for d in train_documents]
        test_documents = [d.copy() for d in test_documents]

        for s in self._steps:
            s.run(train_documents=train_documents,
                           test_documents=test_documents,
                           training_only= training_only)
        print("Pipline End")


def train_ner_pipline():
    train_fold, test_fold = provide_training_data()

    ner_pd = IsolatedPipline(name='mention-extraction-only', steps=[
            pipeline.CrfMentionEstimatorStep(name='crf mention extraction')])
    ner_pd.run(train_documents=train_fold, test_documents=test_fold, training_only=True)


def train_re_pipline():
    train_fold, test_fold = provide_training_data()

    catboost_pd = IsolatedPipline(name='catboost', steps=[pipeline.CatBoostRelationExtractionStep(name='cat-boost relation extraction', use_pos_features=False,
                                                    context_size=2, num_trees=2000, negative_sampling_rate=40.0,
                                                    depth=8, class_weighting=0, num_passes=1)])
    catboost_pd.run(train_documents=train_fold, test_documents=test_fold, training_only=True)


# Using the existing data for testing purposes
def provide_training_data():
    train_folds = [data.loader.read_documents_from_json(f'./jsonl/fold_{i}/train.json') for i in range(5)]
    test_folds = [data.loader.read_documents_from_json(f'./jsonl/fold_{i}/test.json') for i in range(5)]

    return train_folds[0], test_folds[0]


def provide_test_data():
    pass # TODO: Implement logic to receive test data from

def test_ner_pipline():
    pass # TODO: Implement logic to load models and provide predictions for given text

def test_re_pipline():
    pass # TODO: Implement logic to load models and provide predictions for given text

def main():
    train_ner_pipline()

if __name__ == '__main__':
    main()