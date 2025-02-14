import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import anderson, shapiro, f_oneway, ttest_ind, mannwhitneyu, chi2_contingency
import statsmodels.api as sm
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification


class DataInspection:
    def __init__(self):
        """
        Initialize DataInspection class
        """
        self.df = None

    def load_csv(self, file_path):
        """
        Load CSV file
        Args:
            file_path: Path to CSV file
        """
        try:
            self.df = pd.read_csv(file_path)
            print(f"Successfully loaded dataset with {len(self.df)} rows and {len(self.df.columns)} columns")
        except Exception as e:
            print(f"Error loading file: {e}")
            return None

    def _check_data_type(self, column):
        """
        Check data type of column
        Args:
            column: Column name
        Returns:
            str: Data type description
        """
        if pd.api.types.is_numeric_dtype(self.df[column]):
            if self.df[column].nunique() <= 10:
                return "numerical ordinal"
            else:
                return "ratio"
        elif pd.api.types.is_object_dtype(self.df[column]):
            if self.df[column].nunique() <= 10:
                return "non-numeric ordinal"
            else:
                return "nominal"
        return "unknown"

    def column_statistics(self):
        """
        Calculate and display statistics for each column
        """
        for column in self.df.columns:
            print(f"\nColumn name: {column}")
            data_type = self._check_data_type(column)
            print(f"Data type: {data_type}")

            # Calculate central tendency
            if data_type in ["ratio", "numerical ordinal"]:
                print(f"Mean: {self.df[column].mean():.2f}")
                print(f"Median: {self.df[column].median():.2f}")
                print(f"Mode: {self.df[column].mode().iloc[0]}")
                print(f"Standard deviation: {self.df[column].std():.2f}")
                
                # Calculate kurtosis and skewness
                print(f"Kurtosis: {stats.kurtosis(self.df[column]):.2f}")
                print(f"Skewness: {stats.skew(self.df[column]):.2f}")
            else:
                print(f"Mode: {self.df[column].mode().iloc[0]}")
                print(f"Unique values count: {self.df[column].nunique()}")

    def plot_variables(self):
        """
        Display columns available for visualization
        """
        print("\nColumns available for visualization:")
        for i, column in enumerate(self.df.columns, 1):
            print(f"{i}. {column}")

    def plot_distribution(self, column_name):
        """
        Plot distribution
        Args:
            column_name: Column name
        """
        if column_name not in self.df.columns:
            print("Column name does not exist")
            return

        # Handle missing values
        if self.df[column_name].isnull().any():
            print(f"Warning: {column_name} column has {self.df[column_name].isnull().sum()} missing values")
            self.df[column_name].fillna(self.df[column_name].mean() 
                                      if pd.api.types.is_numeric_dtype(self.df[column_name])
                                      else self.df[column_name].mode()[0], 
                                      inplace=True)

        plt.figure(figsize=(10, 6))
        data_type = self._check_data_type(column_name)

        if data_type == "ratio":
            plt.hist(self.df[column_name], bins=30, edgecolor='black')
            plt.title(f'Histogram of {column_name}')
            plt.xlabel(column_name)
            plt.ylabel('Frequency')

        elif data_type == "numerical ordinal":
            sns.boxplot(y=self.df[column_name])
            plt.title(f'Box Plot of {column_name}')

        elif data_type in ["non-numeric ordinal", "nominal"]:
            value_counts = self.df[column_name].value_counts()
            plt.bar(value_counts.index, value_counts.values)
            plt.title(f'Bar Chart of {column_name}')
            plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()
        plt.close()

class DataAnalysis:
    def __init__(self):
        """
        Initialize DataAnalysis class
        """
        self.df = None
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

    def load_csv(self, file_path):
        """
        Load CSV file
        """
        try:
            self.df = pd.read_csv(file_path)
            print(f"Successfully loaded dataset with {len(self.df)} rows and {len(self.df.columns)} columns")
        except Exception as e:
            print(f"Error loading file: {e}")
            return None

    def _check_data_type(self, column):
        """
        Check data type of column
        """
        if pd.api.types.is_numeric_dtype(self.df[column]):
            if self.df[column].nunique() <= 10:
                return "numerical ordinal"
            return "ratio"
        elif pd.api.types.is_object_dtype(self.df[column]):
            if self.df[column].nunique() <= 10:
                return "non-numeric ordinal"
            return "nominal"
        return "unknown"

    def _test_normality(self, data):
        """
        Test data normality
        """
        n = len(data)
        if n > 2000:
            stat, critical_values, sig_level = anderson(data)
            p_value = 0.05  # Anderson test does not provide p-value directly
            is_normal = stat < critical_values[2]  # Using 5% significance level
            test_name = "Anderson-Darling"
        else:
            stat, p_value = shapiro(data)
            is_normal = p_value > 0.05
            test_name = "Shapiro-Wilk"
        
        return test_name, stat, p_value, is_normal

    def _plot_qq(self, data, title):
        """
        Plot Q-Q graph
        """
        plt.figure(figsize=(8, 6))
        stats.probplot(data, dist="norm", plot=plt)
        plt.title(f'{title}Q-Q Plot')
        plt.show()
        plt.close()

    def anova_analysis(self):
        """
        Perform ANOVA analysis
        """
        # Display available variables
        categorical_vars = []
        continuous_vars = []
        
        for col in self.df.columns:
            data_type = self._check_data_type(col)
            if data_type in ["nominal", "non-numeric ordinal"]:
                categorical_vars.append(col)
                print(f"Categorical variable: {col}")
            elif data_type == "ratio":
                continuous_vars.append(col)
                print(f"Continuous variable: {col}")

        # User variable selection
        cat_var = input("\nPlease select categorical variable name: ")
        cont_var = input("Please select continuous variable name: ")

        if cat_var not in categorical_vars or cont_var not in continuous_vars:
            print("Invalid variable selection")
            return

        # Plot Q-Q graph
        self._plot_qq(self.df[cont_var], cont_var)

        # Normality test
        test_name, stat, p_value, is_normal = self._test_normality(self.df[cont_var])
        print(f"\n{test_name} normality test results:")
        print(f"Test statistic: {stat:.4f}")
        print(f"P-value: {p_value:.4f}")
        print(f"Data {'follows' if is_normal else 'does not follow'} normal distribution")

        # Plot boxplot
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=cat_var, y=cont_var, data=self.df)
        plt.title(f'{cont_var} by {cat_var}')
        plt.show()
        plt.close()

        # Select test method based on normality
        groups = [group for _, group in self.df.groupby(cat_var)[cont_var]]
        
        if is_normal:
            # ANOVA test
            f_stat, p_val = f_oneway(*groups)
            print("\nANOVA test results:")
            print(f"F statistic: {f_stat:.4f}")
            print(f"P-value: {p_val:.4f}")
        else:
            # Kruskal-Wallis test
            h_stat, p_val = stats.kruskal(*groups)
            print("\nKruskal-Wallis test results:")
            print(f"H statistic: {h_stat:.4f}")
            print(f"P-value: {p_val:.4f}")

        print(f"\nConclusion: {'Reject' if p_val < 0.05 else 'Failed to reject'} the null hypothesis")
        print(f"Result {'has' if p_val < 0.05 else 'does not have'} statistical significance")
        plt.show()
        plt.close()

    def t_test(self):
        """
        Perform t-test
        """
        # Display available variables
        categorical_vars = []
        continuous_vars = []
        
        for col in self.df.columns:
            data_type = self._check_data_type(col)
            if data_type in ["nominal", "non-numeric ordinal"]:
                categorical_vars.append(col)
                print(f"Categorical variable: {col}")
            elif data_type == "ratio":
                continuous_vars.append(col)
                print(f"Continuous variable: {col}")

        # User variable selection
        cat_var = input("\nPlease select categorical variable name: ")
        cont_var = input("Please select continuous variable name: ")

        if cat_var not in categorical_vars or cont_var not in continuous_vars:
            print("Invalid variable selection")
            return

        # Check if the categorical variable has only two categories
        if self.df[cat_var].nunique() != 2:
            print("Categorical variable must have exactly two categories")
            return

        # Plot Q-Q graph
        self._plot_qq(self.df[cont_var], cont_var)

        # Normality test
        test_name, stat, p_value, is_normal = self._test_normality(self.df[cont_var])
        print(f"\n{test_name} normality test results:")
        print(f"Test statistic: {stat:.4f}")
        print(f"P-value: {p_value:.4f}")
        print(f"Data {'follows' if is_normal else 'does not follow'} normal distribution")

        # Plot boxplot
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=cat_var, y=cont_var, data=self.df)
        plt.title(f'{cont_var} by {cat_var}')
        plt.show()
        plt.close()

        # Get two groups of data
        groups = [group for _, group in self.df.groupby(cat_var)[cont_var]]

        # Select test method based on normality
        if  is_normal:
            # t-test
            t_stat, p_val = ttest_ind(*groups)
            print("\nt-test results:")
            print(f"t statistic: {t_stat:.4f}")
            print(f"P-value: {p_val:.4f}")
        else:
            # Mann-Whitney test
            u_stat, p_val = mannwhitneyu(*groups)
            print("\nMann-Whitney U test results:")
            print(f"U statistic: {u_stat:.4f}")
            print(f"P-value: {p_val:.4f}")

        print(f"\nConclusion: {'Reject' if p_val < 0.05 else 'Failed to reject'} the null hypothesis")
        print(f"Result {'has' if p_val < 0.05 else 'does not have'} statistical significance")
        plt.show()
        plt.close()

    def chi_square_test(self):
        """
        Perform chi-square test
        """
        # Display categorical variables
        categorical_vars = []
        for col in self.df.columns:
            if self._check_data_type(col) in ["nominal", "non-numeric ordinal"]:
                categorical_vars.append(col)
                print(f"Categorical variable: {col}")

        # User variable selection
        var1 = input("\nPlease select the first categorical variable name: ")
        var2 = input("Please select the second categorical variable name: ")

        if var1 not in categorical_vars or var2 not in categorical_vars:
            print("Invalid variable selection")
            return

        # Check category counts
        if self.df[var1].nunique() < 2 or self.df[var2].nunique() < 2:
            print("Each variable must have at least two categories")
            return

        # Create contingency table
        contingency_table = pd.crosstab(self.df[var1], self.df[var2])
        
        # Perform chi-square test
        chi2, p_val, dof, expected = chi2_contingency(contingency_table)

        print("\nChi-square test results:")
        print(f"Chi-square statistic: {chi2:.4f}")
        print(f"P-value: {p_val:.4f}")
        print(f"Degrees of freedom: {dof}")
        
        print(f"\nConclusion: {'Reject' if p_val < 0.05 else 'Failed to reject'} the null hypothesis")
        print(f"Result {'has' if p_val < 0.05 else 'does not have'} statistical significance")
        plt.show()
        plt.close()

    def perform_regression(self):
        """
        Perform regression analysis
        """
        # Display continuous variables
        continuous_vars = []
        for col in self.df.columns:
            if self._check_data_type(col) == "ratio":
                continuous_vars.append(col)
                print(f"Continuous variable: {col}")

        # User variable selection
        independent_var = input("\nPlease select the independent variable name: ")
        dependent_var = input("Please select the dependent variable name: ")

        if independent_var not in continuous_vars or dependent_var not in continuous_vars:
            print("Invalid variable selection")
            return

        # Handle missing values
        data = self.df[[independent_var, dependent_var]].dropna()

        # Check data point count
        print(f"\nValid data points: {len(data)}")

        # Plot Q-Q graph and perform normality test
        for var in [independent_var, dependent_var]:
            self._plot_qq(data[var], var)
            test_name, stat, p_value, is_normal = self._test_normality(data[var])
            print(f"\n{var} {test_name} normality test results:")
            print(f"Test statistic: {stat:.4f}")
            print(f"P-value: {p_value:.4f}")
            print(f"Data {'follows' if is_normal else 'does not follow'} normal distribution")

        # Perform regression analysis
        X = sm.add_constant(data[independent_var])
        model = sm.OLS(data[dependent_var], X).fit()

        print("\nRegression analysis results:")
        print(f"Slope: {model.params[independent_var]:.4f}")
        print(f"Intercept: {model.params['const']:.4f}")
        print(f"R-squared: {model.rsquared:.4f}")
        print(f"P-value: {model.f_pvalue:.4f}")
        print(f"Standard error: {model.bse[independent_var]:.4f}")

        print(f"\nConclusion: {'Reject' if model.f_pvalue < 0.05 else 'Failed to reject'} null hypothesis")
        print(f"Result {'has' if model.f_pvalue < 0.05 else 'does not have'} statistical significance")
        plt.show()
        plt.close()

class SentimentAnalysis:
    def __init__(self, df):
        """
        Initialize sentiment analysis class
        Args:
            df: pandas DataFrame
        """
        self.df = df
        # Initialize VADER analyzer
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Initialize DistilBERT analyzer
        try:
            self.distilbert_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
            self.distilbert_model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
            self.distilbert_pipeline = pipeline("sentiment-analysis", 
                                              model=self.distilbert_model, 
                                              tokenizer=self.distilbert_tokenizer)
        except Exception as e:
            print(f"Warning: DistilBERT model loading failed: {e}")
            self.distilbert_pipeline = None

    def _analyze_vader(self, text):
        """
        Use VADER for sentiment analysis
        """
        scores = self.vader_analyzer.polarity_scores(text)
        
        # Determine sentiment
        if scores['compound'] >= 0.05:
            sentiment = 'Positive'
        elif scores['compound'] <= -0.05:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
            
        print(f"\nVADER analysis results:")
        print(f"Compound score: {scores['compound']:.3f}")
        print(f"Positive score: {scores['pos']:.3f}")
        print(f"Negative score: {scores['neg']:.3f}")
        print(f"Neutral score: {scores['neu']:.3f}")
        print(f"Overall sentiment: {sentiment}")

    def _analyze_textblob(self, text):
        """
        Use TextBlob for sentiment analysis
        """
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Determine sentiment
        if polarity > 0:
            sentiment = 'Positive'
        elif polarity < 0:
            sentiment = 'Negative'
        else:
            sentiment = 'Neutral'
            
        print(f"\nTextBlob analysis results:")
        print(f"Polarity score: {polarity:.3f}")
        print(f"Subjectivity score: {subjectivity:.3f}")
        print(f"Overall sentiment: {sentiment}")

    def _analyze_distilbert(self, text):
        """
        Use DistilBERT for sentiment analysis
        """
        if self.distilbert_pipeline is None:
            print("DistilBERT model not loaded correctly, cannot perform analysis")
            return
            
        try:
            result = self.distilbert_pipeline(text)[0]
            
            print(f"\nDistilBERT Analysis Results:")
            print(f"Predicted Label: {result['label']}")
            print(f"Confidence: {result['score']:.3f}")
            
            # Convert to sentiment labels
            sentiment_map = {
                'POSITIVE': 'Positive',
                'NEGATIVE': 'Negative',
                'NEUTRAL': 'Neutral'
            }
            print(f"Overall Sentiment: {sentiment_map.get(result['label'], 'Unknown')}")
            
        except Exception as e:
            print(f"DistilBERT analysis error: {e}")

    def analyze_sentiment(self):
        """
        Perform sentiment analysis
        """
        # Check if there are text columns
        text_columns = []
        for col in self.df.columns:
            if pd.api.types.is_string_dtype(self.df[col]) and self.df[col].str.len().mean() > 10:
                text_columns.append(col)
        
        if not text_columns:
            print("No suitable text columns found for sentiment analysis")
            return
            
        print("\nAvailable text columns:")
        for i, col in enumerate(text_columns, 1):
            print(f"{i}. {col}")
            
        col_choice = input("Please select a column number to analyze: ")
        try:
            col_index = int(col_choice) - 1
            if col_index < 0 or col_index >= len(text_columns):
                print("Invalid column selection")
                return
            selected_column = text_columns[col_index]
        except ValueError:
            print("Please enter a valid number")
            return
            
        print("\nPlease select sentiment analysis method:")
        print("1. VADER Analysis")
        print("2. TextBlob Analysis")
        print("3. DistilBERT Analysis")
        
        method_choice = input("Please enter option (1-3): ")
        
        # Get text samples for analysis
        sample_size = min(5, len(self.df))
        text_samples = self.df[selected_column].sample(n=sample_size)
        
        for i, text in enumerate(text_samples, 1):
            print(f"\nAnalyzing text {i}:")
            print(f"Text content: {text[:100]}...")  # Only show first 100 characters
            
            if method_choice == '1':
                self._analyze_vader(text)
            elif method_choice == '2':
                self._analyze_textblob(text)
            elif method_choice == '3':
                self._analyze_distilbert(text)
            else:
                print("Invalid choice")
                return

def main():
    """
    Main program control flow
    """
    # Create instances
    inspector = DataInspection()
    analysis = DataAnalysis()
    
    # Request user to load dataset
    file_path = input("Please enter the dataset file path: ")
    try:
        df = pd.read_csv(file_path)
        inspector.df = df
        analysis.df = df
        print(f"Successfully loaded dataset with {len(df)} rows and {len(df.columns)} columns")
    except Exception as e:
        print(f"Failed to load file: {e}")
        return

    while True:
        # Display main menu
        print("\nHow do you want to analyze your data:")
        print("1. Plot variable distribution")
        print("2. Conduct ANOVA")
        print("3. Conduct t-Test")
        print("4. Conduct Chi-square")
        print("5. Conduct Regression")
        print("6. Conduct Sentiment Analysis")
        print("7. QUIT")
        
        # Get user choice
        choice = input("Enter your choice (1-7): ")
        
        # Call corresponding functions based on choice
        if choice == '1':
            inspector.plot_variables()
            column_name = input("Enter the variable name to plot distribution: ")
            inspector.plot_distribution(column_name)
            
        elif choice == '2':
            analysis.anova_analysis()
            
        elif choice == '3':
            analysis.t_test()
            
        elif choice == '4':
            analysis.chi_square_test()
            
        elif choice == '5':
            analysis.perform_regression()
            
        elif choice == '6':
            sentiment_analyzer = SentimentAnalysis(df)
            sentiment_analyzer.analyze_sentiment()
            
        elif choice == '7':
            print("Thank you for using! Goodbye!")
            break
            
        else:
            print("Error: Please enter a number between 1-7")

if __name__ == "__main__":
    main()
