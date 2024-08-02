import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
from sklearn.model_selection import train_test_split

# Load historical data
player_data = pd.read_csv('path/to/player_data.csv')
game_context = pd.read_csv('path/to/game_context.csv')
betting_lines = pd.read_csv('path/to/betting_lines.csv')

# Merge datasets
data = pd.merge(player_data, game_context, on='game_id')
data = pd.merge(data, betting_lines, on=['game_id', 'player_id'])

# Feature engineering
data['moving_avg'] = (
	data.groupby('player_id')['stat']
	.transform(lambda x: x.rolling(window=3).mean())
)
# Add more feature engineering as needed

# Split into features and target
X = data.drop(columns=['actual_outcome'])
y = data['actual_outcome']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
	X, y, test_size=0.2, random_state=42
)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Save the model
joblib.dump(model, 'path/to/your/model.joblib')