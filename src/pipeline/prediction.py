import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging


class PredictionPipeline:
    def __init__(self) -> None:
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')

            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)

            data_preprocessed = preprocessor.transform(features)

            preds = model.predict(data_preprocessed)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:

    def __init__(
        self,
        Marketing_expense: float,
        Production_expense: float,
        Multiplex_coverage: float,
        Budget: float,
        Movie_length: float,
        Lead_Actor_Rating: float,
        Lead_Actress_rating: float,
        Director_rating: float,
        Producer_rating: float,
        Critic_rating: float,
        Trailer_views: int,
        D3_available: str,
        Time_taken: float,
        Twitter_hastags: float,
        Genre: str,
        Avg_age_actors: int,
        Num_multiplex: float
    ) -> None:

        self.Marketing_expense = Marketing_expense
        self.Production_expense = Production_expense
        self.Multiplex_coverage = Multiplex_coverage
        self.Budget = Budget
        self.Movie_length = Movie_length
        self.Lead_Actor_Rating = Lead_Actor_Rating
        self.Lead_Actress_rating = Lead_Actress_rating
        self.Director_rating = Director_rating
        self.Producer_rating = Producer_rating
        self.Critic_rating = Critic_rating
        self.Trailer_views = Trailer_views
        self.D3_available = D3_available
        self.Time_taken = Time_taken
        self.Twitter_hastags = Twitter_hastags
        self.Genre = Genre
        self.Avg_age_actors = Avg_age_actors
        self.Num_multiplex = Num_multiplex

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                'Marketing_expense': [self.Marketing_expense],
                'Production_expense': [self.Production_expense],
                'Multiplex_coverage': [self.Multiplex_coverage],
                'Budget': [self.Budget],
                'Movie_length': [self.Movie_length],
                'Lead_Actor_Rating': [self.Lead_Actor_Rating],
                'Lead_Actress_rating': [self.Lead_Actress_rating],
                'Director_rating': [self.Director_rating],
                'Producer_rating': [self.Producer_rating],
                'Critic_rating': [self.Critic_rating],
                'Trailer_views': [self.Trailer_views],
                'D3_available': [self.D3_available],
                'Time_taken': [self.Time_taken],
                'Twitter_hastags': [self.Twitter_hastags],
                'Genre': [self.Genre],
                'Avg_age_actors': [self.Avg_age_actors],
                'Num_multiplex': [self.Num_multiplex],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
