from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from xgboost import XGBClassifier
from visuals import NAMES, COLORS, style, pitch
P=Path(__file__).parent
st.set_page_config(page_title='Soccer Attacking Phase Classification',page_icon='⚽',layout='wide')
st.markdown("""<style>
.stApp {background:#fff;color:#17251c;}
.block-container{max-width:1150px;padding-top:2rem;}
h1{font-size:2.5rem!important;} h2,h3{color:#17251c!important;}
p,li,[data-testid="stMarkdownContainer"] p{font-size:18px!important;line-height:1.65;}
[data-testid="stCaptionContainer"] p{font-size:15px!important;color:#45574b!important;}
[data-testid="stMetric"]{background:#eef4ef;border:1px solid #c7d5ca;border-radius:8px;padding:18px;}
[data-baseweb="tab"]{font-size:18px!important;min-height:56px;padding:12px 22px!important;}
[data-testid="stForm"]{border:2px solid #b5cabc;padding:20px;border-radius:8px;}
button[kind="primaryFormSubmit"]{min-height:48px;font-size:18px!important;}
</style>""",unsafe_allow_html=True)
@st.cache_resource
def load():
 m=XGBClassifier();m.load_model(P/'model.ubj')
 return m,json.loads((P/'metadata.json').read_text()),pd.read_csv(P/'examples.csv')
m,meta,examples=load();features=meta['features'];classes=meta['classes']
st.caption('Jack Chiang · Soccer tactics research')
st.title('Classifying Soccer Attacking Phases')
st.write('I tested whether player and ball tracking data could identify five types of attack. This app lets you try the model on real phases and see how changing the inputs affects its prediction.')
a,b,c,d=st.columns(4)
a.metric('Attacking phases','2,955');b.metric('Matches','10');c.metric('App model accuracy',f"{meta['accuracy']:.1%}");d.metric('App model macro F1',f"{meta['macro_f1']:.3f}")
st.caption('The app uses a retrained XGBoost model. Results reported in the paper are shown in the Research tab.')
overview,predict,explore,research=st.tabs(['Start here','Try the model','Explore the data','Paper results'])
with overview:
 st.subheader('What does this project do?')
 st.write('A soccer attack can develop in several ways. I trained machine-learning models to recognize five types of attacking phase from measurements of the players and ball.')
 st.write('An attacking phase is a short section of play. The model receives 18 measurements about that section, such as where the ball starts, how far it moves, and how spread out the teams are. It returns one of the five labels below.')
 st.table(pd.DataFrame({'Type of attack':['Build-up','Creation','Direct attack','Fast transition','Finishing'],'Meaning':['Building the attack from farther away from the opponent’s goal.','Working to create a scoring opportunity.','Moving the ball forward quickly.','Attacking soon after winning possession.','The final part of the attack near the opponent’s goal.']}))
 st.subheader('Which tactics were most common?')
 counts=pd.DataFrame({'Tactic':['Creation','Finishing','Build-up','Direct attack','Fast transition'],'Phases':[1226,662,598,337,132]})
 st.plotly_chart(style(px.bar(counts,x='Phases',y='Tactic',orientation='h',text='Phases',color_discrete_sequence=['#285943'],title='All 2,955 phases in the paper')).update_layout(yaxis=dict(autorange='reversed')),width='stretch')
 st.caption('Source: paper, Dataset section. Fast transitions account for only 132 of the 2,955 phases.')
 st.subheader('How the study was tested')
 split=pd.DataFrame({'Set':['Training · 8 matches','Testing · 2 matches'],'Phases':[2388,567]})
 st.plotly_chart(style(px.bar(split,x='Set',y='Phases',text='Phases',color_discrete_sequence=['#2563a0'],title='Entire matches were kept separate')),width='stretch')
 st.subheader('What I found')
 st.write('In my paper, Enhanced XGBoost classified 91.9% of test phases correctly—about 92 out of 100. Enhanced Random Forest scored best when all five tactics were weighted equally. Fast transitions were hardest to identify, partly because there were only 132 examples in the dataset.')
 st.write('The study used 2,955 phases from 10 Australian A-League matches. Eight matches were used to train the models, and two different matches were used to test them.')
 st.info('To try it: open Try the model, pick an example, and look at the prediction. You can change the measurements and press Predict tactic to see what happens. No upload is needed.')
 st.caption('The app uses numerical tracking features, not video. Its labels come from SkillCorner’s existing system. It identifies the type of attack; it does not choose a tactic for a coach.')
with predict:
 st.subheader('Change a phase. See what the model predicts.')
 st.write('Start with a real example. Move the sliders to explore a different version of that attack. No files or coding needed.')
 label=st.selectbox('Start with this type of attack',classes,format_func=lambda v:NAMES[v])
 subset=examples[examples.tactic_label==label]
 idx=st.selectbox('Choose an example',subset.index.tolist(),format_func=lambda i:f"Match {int(examples.loc[i,'match_id'])} · phase {int(examples.loc[i,'phase_index'])} · minute {examples.loc[i,'match_minute']:.1f}")
 original=examples.loc[idx,features].astype(float); values=original.copy()
 left,right=st.columns([1,1.25])
 fields=[('ball_x_start','Ball starting position (meters)','Zero is midfield; values use the original tracking coordinate system.'),('ball_progress_m','Ball movement along the field (meters)','Signed change in ball x position, not total distance traveled.'),('duration','Length of the phase (seconds)','Changing this also updates progress speed.'),('attack_width_mean','Attacking team width (meters)','Average side-to-side spread of the attacking team.'),('attackers_ahead_of_ball_mean','Attackers ahead of the ball','Average count across the phase, so fractions are possible.')]
 with left:
  if st.button('Reset this example'):
   for f,_,_ in fields: st.session_state.pop(f'slider_{idx}_{f}',None)
  for f,title,help_text in fields:
   low=float(examples[f].min());high=float(examples[f].max());v=float(original[f])
   values[f]=st.slider(title,min_value=low,max_value=high,value=v,key=f'slider_{idx}_{f}',help=help_text)
 values['progress_speed_mps']=values['ball_progress_m']/values['duration'] if values['duration']>0 else 0.
 with right:
  st.plotly_chart(pitch(values),width='stretch')
  st.caption('Schematic: shows only ball movement along the x-axis. The centered line is not the real sideways path; the field outline is illustrative. Player positions cannot be recovered from phase averages.')
 x=pd.DataFrame([values],columns=features);probs=m.predict_proba(x)[0];baseline=m.predict_proba(pd.DataFrame([original],columns=features))[0]
 winner=classes[int(probs.argmax())]
 st.subheader('Prediction: '+NAMES[winner])
 st.caption('Original example label: '+NAMES[label]+'. The other model inputs stay at their original values. Slider changes are hypothetical, and some combinations may be unrealistic.')
 chart=pd.DataFrame({'Tactic':[NAMES[c] for c in classes]*2,'Probability':list(baseline)+list(probs),'Version':['Original example']*5+['Your changes']*5})
 fig=style(px.bar(chart,x='Probability',y='Tactic',color='Version',barmode='group',orientation='h',color_discrete_map={'Original example':'#b7c8be','Your changes':'#285943'},range_x=[0,1],title='How the prediction changes'))
 fig.update_xaxes(tickformat='.0%');fig.update_traces(hovertemplate='%{y}: %{x:.1%}<extra>%{fullData.name}</extra>')
 st.plotly_chart(fig,width='stretch')
 st.caption('Scores are model probabilities, not guarantees. This identifies a type of attack; it does not decide which tactic a coach should use.')
 with st.expander('See all 18 inputs'):
  st.dataframe(pd.DataFrame({'Original':original,'Your changes':values}),width='stretch')
with explore:
 st.subheader('What does the data look like?')
 st.write('Each dot below is one attacking phase from the two held-out test matches. Hover over it to see the match and phase.')
 view=examples.copy();view['Tactic']=view.tactic_label.map(NAMES)
 selected=st.multiselect('Show tactics',list(NAMES.values()),default=list(NAMES.values()))
 view=view[view.Tactic.isin(selected)]
 st.caption(f'{len(view):,} of 567 test phases shown. These plots do not include the training matches.')
 if len(view):
  axes={'Ball starting position (m)':'ball_x_start','Ball progress (signed m)':'ball_progress_m','Duration (seconds)':'duration','Attacking width (m)':'attack_width_mean','Defenders within 5 m':'defenders_within_5m_mean'}
  a,b=st.columns(2)
  xf=a.selectbox('Horizontal axis',list(axes),index=0);yf=b.selectbox('Vertical axis',list(axes),index=1)
  st.plotly_chart(style(px.scatter(view,x=axes[xf],y=axes[yf],color='Tactic',color_discrete_map=COLORS,labels={axes[xf]:xf,axes[yf]:yf},hover_data=['match_id','phase_index'],opacity=.7,title='Tracking measurements by tactic')),width='stretch')
  st.plotly_chart(style(px.box(view,x='Tactic',y='duration',color='Tactic',color_discrete_map=COLORS,labels={'duration':'Phase duration (seconds)'},title='How long does each type of phase last?')).update_layout(showlegend=False),width='stretch')
  match=st.selectbox('View one match',sorted(view.match_id.unique()),format_func=lambda v:f'Match {v}')
  timeline=view[view.match_id==match].sort_values('match_minute')
  st.plotly_chart(style(px.scatter(timeline,x='match_minute',y='Tactic',color='Tactic',color_discrete_map=COLORS,hover_data=['phase_index','duration'],labels={'match_minute':'Match minute'},title='Attacking phases through the match')).update_layout(showlegend=False),width='stretch')
 else: st.info('Choose at least one tactic to show its phases.')
with research:
 st.subheader('My research')
 st.write('Machine Learning-Based Classification of Soccer Attacking Phases Using Spatiotemporal Tracking Features')
 st.write('This project uses SkillCorner open tracking data from 10 Australian A-League matches to distinguish build-up, creation, direct attack, fast transition, and finishing.')
 st.write('The labels came from SkillCorner’s AI-based Phases of Play system. My model attempts to reproduce those labels using a smaller set of tracking features; they are not independent labels created by coaches for this project.')
 st.subheader('Results reported in the paper')
 results=pd.DataFrame({'Model':['Original Random Forest','Enhanced Random Forest','Original XGBoost','Enhanced XGBoost','SVC','MLP','KNN'],'Accuracy':[.912,.912,.914,.919,.879,.858,.756],'Macro F1':[.863,.873,.870,.864,.825,.790,.649]})
 metric=st.radio('Compare models using',['Accuracy','Macro F1'],horizontal=True)
 st.caption('Accuracy counts all correct predictions. Macro F1 gives each tactic equal weight and balances precision with recall.')
 ordered=results.sort_values(metric)
 fig=style(px.bar(ordered,x=metric,y='Model',orientation='h',text=metric,range_x=[0,1],color_discrete_sequence=['#285943'],title='Model comparison · reported in the paper'))
 fig.update_traces(texttemplate='%{x:.1%}' if metric=='Accuracy' else '%{x:.3f}')
 if metric=='Accuracy': fig.update_xaxes(tickformat='.0%')
 st.plotly_chart(fig,width='stretch')
 f1=pd.DataFrame({'Tactic':list(NAMES.values()),'F1':[.98,.92,.91,.59,.91]})
 st.plotly_chart(style(px.bar(f1,x='Tactic',y='F1',text='F1',range_y=[0,1],color_discrete_sequence=['#2563a0'],title='Enhanced XGBoost: F1 for each tactic · paper results')),width='stretch')
 st.caption('Source: Results and Discussion. These are the original paper’s scores, not scores recalculated for the app model.')
 st.write('Enhanced XGBoost had the highest accuracy (91.9%). Enhanced Random Forest had the highest macro F1 (0.873), which gives each tactic equal weight. Fast transition was the hardest category: the paper reports 13 correct predictions out of 25 test examples.')
 st.subheader('Model running in this app')
 st.write(f"The deployed XGBoost model was retrained from the supplied notebook configuration on {meta['train_phases']:,} phases. Matches 2015213 and 1899585 were held out entirely for evaluation ({meta['test_phases']:,} phases). The scores above were recomputed for this deployed model.")
 st.write('Read the grid by row: the row is the supplied label, and the column is the model’s prediction. Numbers on the diagonal are correct predictions.')
 st.plotly_chart(style(px.imshow(meta['confusion'],x=[NAMES[c] for c in classes],y=[NAMES[c] for c in classes],text_auto=True,labels={'x':'Predicted tactic','y':'Reference tactic','color':'Phases'},color_continuous_scale='Greens',title='App model: predictions on 567 test phases')),width='stretch')
 importance=pd.DataFrame({'Feature':[f.replace('_',' ').capitalize() for f in features],'Importance':m.feature_importances_}).sort_values('Importance').tail(10)
 st.plotly_chart(style(px.bar(importance,x='Importance',y='Feature',orientation='h',color_discrete_sequence=['#a67416'],title='App model: ten highest feature importance scores')),width='stretch')
 st.caption('XGBoost feature importance measures how the trained trees use each input. It does not establish cause and effect or explain one specific prediction.')
 st.write('The 18 features describe ball position and progress, attacking and defending shape, nearby defenders, phase timing, and progress speed. Match and phase identifiers are excluded from training inputs.')
 st.subheader('Limitations and next steps')
 st.write('The study covers 10 matches from one league. Averaging features across each phase loses the order of player and ball movements. More matches, other leagues, and sequence-based models would help test how well the approach generalizes.')
 st.caption('Based on the local paper draft TS8_ Final Research Paper Task Sheet-4.pdf, sections 2–7.')
 st.markdown('[SkillCorner open data](https://github.com/SkillCorner/opendata)')
