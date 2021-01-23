

#%%
import pandas as pd
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

business = pd.read_json('yelp_business.json', lines=True, nrows=18000)
reviews = pd.read_json('yelp_review.json', lines=True, nrows=18000)
photos = pd.read_json('yelp_photo.json', lines=True, nrows=18000)
tips = pd.read_json('yelp_tip.json', lines=True, nrows=18000)
users = pd.read_json('yelp_user.json', lines= True, nrows=18000)
checkins = pd.read_json('yelp_checkin.json', lines=True, nrows=18000)

pd.options.display.max_columns = 60
pd.options.display.max_colwidth = 500

# print(reviews.columns)
# print("there are " , len(reviews.columns) , " columns for reviews")
# print(business.columns)
# print(len(business.columns))

# print(users.describe())

merged_data = pd.merge(business, reviews, how='left', on='business_id' )
merged_data =  pd.merge(merged_data, photos, how='left', on='business_id')
merged_data =  pd.merge(merged_data, users, how='left', on='business_id')
merged_data =  pd.merge(merged_data, tips, how='left', on='business_id')
merged_data =  pd.merge(merged_data, checkins, how='left', on='business_id')

print(merged_data.columns)

# for column in merged_data.columns:
#   print(column, " is of type: ", type(column))

features_to_remove = ['address', 'attributes', 'business_id', 'categories', 'city', 'hours', 'is_open', 'latitude', 'longitude', 'name', 'neighborhood', 'postal_code', 'state', 'time']
merged_data.drop(labels=features_to_remove, axis=1, inplace=True)

check_nans = merged_data.isna().any()

merged_data.fillna({
    'average_review_age':0,             
    'average_review_length':0,          
    'average_review_sentiment':0,       
    'number_funny_votes':0,             
    'number_cool_votes':0,              
    'number_useful_votes':0,            
    'average_caption_length':0,         
    'number_pics':0,                    
    'average_number_friends':0,         
    'average_days_on_yelp':0,           
    'average_number_fans':0,            
    'average_review_count':0,           
    'average_number_years_elite':0,     
    'average_tip_length':0,             
    'number_tips':0,                    
    'weekday_checkins':0,              
    'weekend_checkins':0, }, inplace=True)
merged_data.isna().any()

merged_data.corr()


plt.scatter(merged_data.stars, merged_data.average_review_sentiment, alpha=0.5)
plt.xlabel('rating')
plt.show()
plt.scatter(merged_data.stars, merged_data.average_review_length, alpha=0.8)
plt.show()

features = merged_data[['average_review_length', 'average_review_sentiment']]
ratings = merged_data['stars']



X_train, X_test, y_train, y_test =  train_test_split(features, ratings, test_size=0.2, random_state=1)



model = LinearRegression()
model.fit(X_train, y_train)

model.score(X_train, y_train)
model.score(X_test, y_test)

y_predicted = model.predict(X_test)

plt.scatter(y_test, y_predicted)
plt.xlabel('yelp rating')
plt.ylabel('predicted rating')
plt.ylim(1,5)
plt.show()

for prediction in y_predicted:
  print(predicion)

# %%
