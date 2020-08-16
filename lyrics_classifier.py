import pandas as pd
import json
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
import jsonlines
from sklearn.naive_bayes import MultinomialNB

warnings.filterwarnings('ignore')

list_artists = ['Queen', 'Muse', 'Janelle Monáe', 'Hot Chip',
                'LCD Soundsystem', 'The Postal Service',
                'Daft Punk', 'The Strokes']

df = pd.read_json('lyrics.jl', lines=True)
df_fix = df[df.titles != '']
for artist in list_artists:
        df_fix.loc[df_fix.artists.str.contains(artist), 'main_artist']=artist

df_fix = df_fix.dropna()

df_fix = df_fix.drop_duplicates(subset = 'lyrics')
df_fix.titles.value_counts().head(20)
df_fix[df_fix['titles']=='Invincible']
df_nodupe = df_fix.drop_duplicates(subset = ['titles', 'main_artist'])
df_nodupe.loc[:,'lyrics'] = df_nodupe['lyrics'].str.replace('\r\n', ' ') #could probably do this better with regex
df_nodupe.loc[:,'lyrics'] = df_nodupe['lyrics'].str.replace('\n', ' ')
df_nodupe.drop_duplicates(subset = 'lyrics')
df_nodupe.dropna(inplace=True)
df_nodupe['first_6'] = df_nodupe.titles.str[0:8]
df_nodupe.first_6 = df_nodupe.first_6.str.lower()
df_nodupe_title = df_nodupe.drop_duplicates(subset = ['first_6', 'main_artist'])
df_nodupe_title[df_nodupe_title.first_6.str.contains('how')]

X = df_nodupe_title.lyrics
y = df_nodupe_title.main_artist
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

cv = CountVectorizer(lowercase=True, stop_words='english', token_pattern='[a-z]+')
cv.fit(X_train)
X_cv = cv.transform(X_train)
X_test_cv = cv.transform(X_test)

X_cv.todense()

nb = MultinomialNB()

nb.fit(X_cv, y_train)
y_pred = nb.predict(X_test_cv)
nb.fit(X_tf, y_train)
y_pred_tfid = nb.predict(X_test_tf)

predictions_cv = nb.predict_proba(X_test_cv)
import seaborn as sns
sns.heatmap(predictions_cv)

def bayes_eval(y_true, y_pred, listofartists):
    cv_confusionmatrix = confusion_matrix(y_true, y_pred, labels = listofartists ,normalize = 'true' )
    ax = sns.heatmap(cv_confusionmatrix, annot = True, xticklabels = listofartists, yticklabels = listofartists)
    ax.set(xlabel='Predicted', ylabel='True')
    print(classification_report(y_true, y_pred))

import spacy
model = spacy.load('en_core_web_md')
X_tokens = [model(song) for song in X]

lemmatized_word = []
lemmatized_song = ''
X_lemmatized = []
for song in X_tokens:
    lemmatized_word = []
    for word in song:
        lemmatized_word.append(word.lemma_)
    lemmatized_song = ' '.join(lemmatized_word)
    X_lemmatized.append(lemmatized_song)
X_lemmatized[5]

X_train_lem, X_test_lem, y_train, y_test = train_test_split(X_lemmatized, y, test_size=0.33, random_state=42)
cv_lem = CountVectorizer(stop_words='english', lowercase = False, tokenizer = None, token_pattern='[a-z]+')
cv_lem.fit(X_train_lem)
X_train_lemcv = cv_lem.transform(X_train_lem)
X_test_lemcv = cv_lem.transform(X_test_lem)
cv_lem.get_feature_names()==cv.get_feature_names()


# In[56]:


nb.fit(X_train_lemcv, y_train)
y_pred_lemcv = nb.predict(X_test_lemcv)
bayes_eval(y_test, y_pred_lemcv, list_artists) #slightly different


# In[57]:


cv_lem = CountVectorizer(stop_words='english', lowercase = False, tokenizer = None, token_pattern='[a-z]+', min_df = 3)
cv_lem.fit(X_train_lem)
X_train_lemcv = cv_lem.transform(X_train_lem)
X_test_lemcv = cv_lem.transform(X_test_lem)
cv_lem.get_feature_names()==cv.get_feature_names()

nb.fit(X_train_lemcv, y_train)
y_pred_lemcv = nb.predict(X_test_lemcv)
bayes_eval(y_test, y_pred_lemcv, list_artists) #changing min_df to 3 really improved things


# In[58]:


bayes_eval(y_test, y_pred_lemcv, list_artists) #changing min_df to 3


# In[59]:


vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', token_pattern='[a-z]+', min_df= 5)
X_train_lemtf =vectorizer.fit_transform(X_train_lem)
X_test_lemtf = vectorizer.transform(X_test_lem)


# In[60]:


nb.fit(X_train_lemtf, y_train)
y_pred_lemtf = nb.predict(X_test_lemtf)
bayes_eval(y_test, y_pred_lemtf, list_artists) #still really really bad


# In[ ]:





# In[ ]:





# # oversampling and undersampling

# In[61]:


#undersample first
from imblearn.under_sampling import RandomUnderSampler, NearMiss


# In[62]:


list_artists


# In[63]:


df_nodupe_title.groupby('main_artist').count()


# In[95]:


samp_dict = {'Queen':50, 'Muse':50, 'Janelle Monáe':30, 'Hot Chip':30,
             'LCD Soundsystem':30, 'The Postal Service':6, 'Daft Punk':30, 'The Strokes':38 }
rus = RandomUnderSampler(random_state=10, sampling_strategy=samp_dict)
nm = NearMiss(sampling_strategy=samp_dict)


# In[96]:


X_rus, y_rus = rus.fit_resample(X_train_lemcv, y_train)
X_nm, y_nm = nm.fit_resample(X_train_lemcv, y_train)


# In[97]:


X_rus.shape, y_rus.shape, np.unique(y_rus, return_counts=True)


# In[67]:


X_nm.shape, y_nm.shape, np.unique(y_nm, return_counts=True) #decreased overall amount of songs by about half


# In[68]:


X_train_lemcv.shape


# In[ ]:


nb.fit(X_rus, y_rus)
y_pred_rus = nb.predict(X_test_lemcv)


# In[69]:


bayes_eval(y_test, y_pred_rus, list_artists) #rus destroyed accuracy - and queen can't be guessed as queen


# In[ ]:


nb.fit(X_nm, y_nm)
y_pred_nm = nb.predict(X_test_lemcv)


# In[70]:


bayes_eval(y_test, y_pred_nm, list_artists) # near miss undersample


# In[71]:


# try oversampling instead


# In[72]:


from imblearn.over_sampling import RandomOverSampler, SMOTE
upsample_dict = {'Queen':144, 'Muse':75, 'Janelle Monáe':50, 'Hot Chip':64, 'LCD Soundsystem':50,
                 'The Postal Service':20, 'Daft Punk':50, 'The Strokes':50 }

ros = RandomOverSampler(random_state=10)


# In[73]:


X_ros, y_ros = ros.fit_resample(X_train_lemcv, y_train)


# In[74]:


np.unique(y_train, return_counts=True)


# In[75]:


np.unique(y_ros, return_counts=True)


# In[ ]:


nb.fit(X_ros, y_ros)
y_pred_ros = nb.predict(X_test_lemcv)


# In[76]:


bayes_eval(y_test, y_pred_ros, list_artists) #ros upsample to 144 for each


# In[77]:


from imblearn.over_sampling import SMOTE


# In[78]:


sm = SMOTE(random_state=42)


# In[79]:


X_sm, y_sm = sm.fit_resample(X_train_lemcv, y_train)


# In[80]:


np.unique(y_sm, return_counts=True)


# In[ ]:


nb.fit(X_sm, y_sm)
y_pred_sm = nb.predict(X_test_lemcv)


# In[81]:


#SMOTE upsample to 144 for each
bayes_eval(y_test, y_pred_sm, list_artists) #works better for me than random


# In[82]:


#combine upsample and down sample


# In[83]:


from imblearn.combine import SMOTEENN


# In[84]:


samp_dict


# In[85]:


sme = SMOTEENN(random_state=42)


# In[86]:


X_sme, y_sme = sme.fit_resample(X_train_lemcv, y_train)


# In[87]:


np.unique(y_sme, return_counts=True)


# In[ ]:


nb.fit(X_sme, y_sme)
y_pred_sme = nb.predict(X_test_lemcv)


# In[88]:


bayes_eval(y_test, y_pred_sme, list_artists) #that really killed accuracy, why did it bring the queen songs down to 2?


# In[89]:


from imblearn.combine import SMOTETomek
upsample_dict


# In[126]:


smt_dict = {'Queen': 144,'Muse': 100,'Janelle Monáe': 60, 'Hot Chip': 64, 'LCD Soundsystem': 60,'The Postal Service': 30,'Daft Punk': 60,'The Strokes': 60}
smt = SMOTETomek(random_state=42 )
sampling_strategy=smt_dict
X_smt, y_smt = smt.fit_resample(X_train_lemcv, y_train)
nb.fit(X_smt, y_smt)
y_pred_smt = nb.predict(X_test_lemcv)


# In[128]:


bayes_eval(y_test, y_pred_smt, list_artists) #SMOTETomek looks the same as SMOTE - 144 songs for each


# In[ ]:





# In[91]:


# if time, write function to optimize the upsampling


# In[92]:


# make word clouds for fun


# In[93]:


from matplotlib import pyplot as plt
import wordcloud

#mask = np.____((500, ____, 3), _____)
#mask[150:350,____:350,:] = 255  # masked out area
def make_wordclouds(all_songs_list):
    fig, axs = plt.subplots(nrows=4, ncols=2,figsize=(10, 10))
    axs = axs.flatten()
    list_artists = ['Queen', 'Muse', 'Janelle Monáe', 'Hot Chip', 'LCD Soundsystem','The Postal Service', 'Daft Punk', 'The Strokes']
#     axes_list =[ ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]
    for i, artist_list in enumerate(allsongs_list):
        cloud = wordcloud.WordCloud(background_color="white",
                max_words=50,
                collocations=True,  # calculates frequencies
                contour_color='steelblue').generate(''.join(artist_list))
                # stop words are removed!
        axs[i].imshow(cloud, interpolation='bilinear')
        axs[i].axis('off')
        name = str(artist_list)
        axs[i].set_title(str(list_artists[i]))

plt.show()


# In[122]:


make_wordclouds(allsongs_list)


# In[ ]:





# In[99]:


#trying a random forest classifier for fun
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()


# In[100]:


rfc.fit(X_train_lemcv, y_train)
y_pred_rfc = rfc.predict(X_test_lemcv)


# In[101]:


bayes_eval(y_test, y_pred_rfc, list_artists) #randomforest did a little worse than the bayes


# In[102]:


samp_dict


# In[103]:


upsample_dict


# In[104]:


downsam_dict = {'Queen': 100,
 'Muse': 75,
 'Janelle Monáe': 50,
 'Hot Chip': 64,
 'LCD Soundsystem': 50,
 'The Postal Service': 20,
 'Daft Punk': 40,
 'The Strokes': 50}


# In[105]:


np.unique(y_train, return_counts=True)


# In[106]:


from imblearn.under_sampling import EditedNearestNeighbours
from imblearn.pipeline import make_pipeline
upsmote= SMOTE(random_state=42, sampling_strategy= upsample_dict)
enn = EditedNearestNeighbours() # this works poorly in the pipeline
rus = RandomUnderSampler(random_state=42, sampling_strategy=downsam_dict)


# In[107]:


up_down_pipeline = make_pipeline(upsmote, rus, nb)


# In[108]:


up_down_pipeline.fit(X_train_lemcv, y_train)


# In[109]:


y_pred_pipeline = up_down_pipeline.predict(X_test_lemcv)


# In[110]:


bayes_eval(y_test, y_pred_pipeline, list_artists) # Is this better? Unclear


# In[111]:


#try pipeline with tfidf vectorized data
up_down_pipeline.fit(X_train_lemtf, y_train)


# In[112]:


y_pred_pipeline_tf = up_down_pipeline.predict(X_test_lemtf)


# In[113]:


bayes_eval(y_test, y_pred_pipeline, list_artists) #with resampling, looks essentially the same was with cv


# In[124]:


accuracy_summary = {'Strategy':['CV', 'Tfidf', 'CV+lemma', 'CV+lemma+min_df','Tfidf+lemma', 'Tfidf+lemma+min_df',
                   'CV+lemma+Rus',
                   'CV+lemma+NearMiss', 'CV+lemma+Ros', 'CV+lemma+SMOTE', 'CV+lemma+smoteteen',
                   'CV+lemma+Smotetomek','Random Forest - CV+lemma', 'CV+lemma+pipeline',
                   'Tfidf+lemma+pipeline'], 'Accuracy':[0.47, 0.28, 0.55, 0.55, 0.28,
                                                                                    0.29, 0.44, 0.42, 0.49,0.56,
                                                                                    0.35, 0.53, 0.45, 0.51, 0.51]}
df_summary = pd.DataFrame(accuracy_summary, columns = ['Strategy', 'Accuracy'])


# In[125]:


df_summary.sort_values('Accuracy', ascending = False)


# In[123]:


make_wordclouds(allsongs_list)


# ### What I've learned from doing this:
# - Lemmatization makes a big difference
# - Several issues with really small sample sizes
#     - in training
#     - in calculating accuracy
# - Pipelines/resampling/etc.
# ### Further questions:
# - How to more effectively use Spacy
# - How can we use Spacy to effectively look at document similarity
# - Upsampling before tfidf?
# ### Things to add to this project:
# - Combine into single py file with the best looking model, allow user input of new song for test (in progress)
# - Iteratively remove artists to see what the best combination is
# - Iterirate through different values for the sampling strategies to optimize pipeline
# - Add features - usage of parts of speech, sentiment analysis
# - Mask word clouds onto some kind of symbol/art for the artist
# - Add Spotify playlist

# In[116]:


get_ipython().system('dbus-send --print-reply --dest=org.mpris.MediaPlayer2.spotify /org/mpris/MediaPlayer2 org.mpris.MediaPlayer2.Player.Play')



# In[117]:


#try the spacy with spacys own vectorization
type(X_tokens[1])


# In[118]:


#turn each word into a vector
def vector(tokens):

    song_vectors = []
    for item in tokens:
        word_vectors=[]
        for word in item:
            word_vectors.append(model.vocab[word].vector)
    song_vectors.append(word_vectors)
    return song_vectors


# In[119]:


#totally confused
spacy_vectors = vector(X_tokens)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[120]:


from spacy.lemmatizer import Lemmatizer
from spacy.lookups import Lookups

lookups = Lookups()
lemmatizer = Lemmatizer(lookups)
X_lemmas = lemmatizer(X)
print(X_lemmas)

#this is a dead end - lemmatizer is only for single words maybe?


# In[121]:


# make lists for each artist
# maybe this was unecessary

queen_list = df_nodupe_title[df_nodupe_title['main_artist']=='Queen'].lyrics.to_list()
muse_list = df_nodupe_title[df_nodupe_title['main_artist']=='Muse'].lyrics.to_list()
janelle_list = df_nodupe_title[df_nodupe_title['main_artist']=='Janelle Monáe'].lyrics.to_list()
hotchip_list = df_nodupe_title[df_nodupe_title['main_artist']=='Hot Chip'].lyrics.to_list()
lcd_list = df_nodupe_title[df_nodupe_title['main_artist']=='LCD Soundsystem'].lyrics.to_list()
postalservice_list = df_nodupe_title[df_nodupe_title['main_artist']=='The Postal Service'].lyrics.to_list()
daftpunk_list = df_nodupe_title[df_nodupe_title['main_artist']=='Daft Punk'].lyrics.to_list()
strokes_list = df_nodupe_title[df_nodupe_title['main_artist']=='The Strokes'].lyrics.to_list()

allsongs_list = [queen_list, muse_list, janelle_list, hotchip_list, lcd_list, postalservice_list, daftpunk_list,strokes_list]

#there has got to be a better way to do this, but can't change the list name in a for loop

#for artist in list_artists:
   #lyrics_dict = df_nodupe_title[df_nodupe_title['main_artist']==str(artist)].lyrics.to_dict()

#df_nodupe_title.index = df_nodupe_title.main_artist
#lyrics_dict = df_nodupe_title.to_dict('index')# dictionary is overwriting



# In[ ]:





# In[ ]:





# In[ ]:
