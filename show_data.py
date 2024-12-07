from datasets import load_dataset
import pandas as pd

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)

# Load AG News dataset
ag_news = load_dataset("ag_news")

# Display key columns for AG News dataset
print("AG News Dataset:")
print("Columns:", ag_news['train'].column_names)
print("\nSample rows from key columns:")
print(ag_news['train'].select(range(2)).to_pandas()[['label', 'text']])  # Display 'label' and 'text' columns

print("\n" + "="*50 + "\n")

# Load SQuAD dataset
poetry = load_dataset("merve/poetry")

# Display key columns for SQuAD dataset
print("poetry Dataset:")
print("Columns:", poetry['train'].column_names)
print("\nSample rows from key columns:")
print(poetry['train'].select(range(2)).to_pandas()[['poem name', 'content', ]])  # Display 'context', 'question', and 'answers' columns
