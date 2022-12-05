'''
Created by Mahdi Mashayekhi
Email : MahdiMashayekhi.ai@gmail.com
Github : https://github.com/MahdiMashayekhi-AI
Site : http://mahdimashayekhi.gigfa.com
YouTube : https://youtube.com/@MahdiMashayekhi
Twitter : https://twitter.com/Mashayekhi_AI
LinkedIn : https://www.linkedin.com/in/mahdimashayekhi/

'''
'''
Predicting the reach of Instagram posts is one of the most important tasks for every business that relies heavily on social media customers. So in such competition, it is very important to know how the Instagram algorithm works. In this article, Im going to walk you through an implementation of the Instagram algorithm with Machine Learning using Python to understand how your posts can get more reach on Instagram.

'''

# Import libraries
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor

# Read dataset
data = pd.read_csv("instagram_reach.csv")

# Display information about the dataset
data.info()

stopwords = set(STOPWORDS) 
stopwords.add('will')
sns.set()
plt.style.use('seaborn-whitegrid') # Optional

def WordCloudPlotter(dfColumn):
    colData = data[dfColumn]
    textCloud = ''
    
    for mem in colData:
        textCloud = textCloud + str(mem)
    
    # plotting word cloud
    wordcloud = WordCloud(width = 800, height = 800,background_color ='white', 
                          stopwords = stopwords,  min_font_size = 10).generate(textCloud)
    plt.figure(figsize = (8, 8), facecolor = None) # Create figure
    plt.style.use('seaborn-whitegrid') #Optional
    plt.imshow(wordcloud) # Display image
    plt.rcParams.update({'font.size': 25})
    plt.axis("off") 
    plt.title('Word Cloud: ' + str(dfColumn)) # Title window
    plt.tight_layout(pad = 0) 
  
    plt.show() # Display
    
WordCloudPlotter('Caption') # Search for Caption

WordCloudPlotter('Hashtags') # Search for Hashtags

def PlotData(features):
    plt.figure(figsize= (20, 10)) # Create figure
    pltNum = 1
    for mem in features:
        plt.subplot(1, 2 , pltNum) # Size of thr figure
        plt.style.use('seaborn-whitegrid') # Optional
        plt.grid(True) # Optional
        plt.title('Regplot Plot for '+ str(mem)) # Title window
        sns.regplot(data = data, x = mem, y = 'Likes' , color = 'green')
        pltNum += 1
    
    plt.show()
    
PlotData(['Followers', 'Time since posted'])


features = np.array(data[['Followers', 'Time since posted']], dtype = 'float32')
targets = np.array(data['Likes'], dtype = 'float32')
maxValLikes = max(targets)
print('Max value of target is {}'.format(maxValLikes))

targets = targets/maxValLikes

# Split the Data into Train and Test
xTrain, xTest, yTrain, yTest = train_test_split(features, targets, test_size = 0.1, random_state = 42)

# Normalization
stdSc = StandardScaler()
xTrain = stdSc.fit_transform(xTrain)
xTest = stdSc.transform(xTest)


gbr = GradientBoostingRegressor()
gbr.fit(xTrain, yTrain)

predictions = gbr.predict(xTest)
plt.scatter(yTest, predictions)
plt.style.use('seaborn-whitegrid')
plt.xlabel('true values')
plt.ylabel('predicted values')
plt.title('GradientRegressor')
plt.plot(np.arange(0,0.4, 0.01), np.arange(0, 0.4, 0.01), color = 'green')
plt.grid(True)

def PredictFollowers(model, followerCount, scaller, maxVal):
    followers = followerCount * np.ones(24)
    hours = np.arange(1, 25)
    
    # Defining vector 
    featureVector = np.zeros((24, 2))
    featureVector[:, 0] = followers
    featureVector [:, 1] = hours
    
    # Doing scalling
    featureVector = scaller.transform(featureVector)
    predictions = model.predict(featureVector)
    predictions = (maxValLikes * predictions).astype('int')
    
    plt.figure(figsize= (10, 10))
    plt.plot(hours, predictions)
    plt.style.use('seaborn-whitegrid')
    plt.scatter(hours, predictions, color = 'g')
    plt.grid(True)
    plt.xlabel('hours since posted')
    plt.ylabel('Likes')
    plt.title('Likes progression with ' + str(followerCount) +' followers')
    plt.show()

# likes progression for 100 followers
PredictFollowers(gbr, 100, stdSc, maxValLikes)

# likes progression for 200 followers
PredictFollowers(gbr, 200, stdSc, maxValLikes)

# Like progression for 1000 followers
PredictFollowers(gbr, 1000, stdSc, maxValLikes)