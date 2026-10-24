from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from xgboost import XGBClassifier
from visuals import NAMES, COLORS, style

P=Path(__file__).parent
st.set_page_config(page_title='Soccer Attacking Phases | Jack Chiang',page_icon='⚽',layout='wide')
st.markdown('''<style>
.block-container{max-width:1050px;padding-top:4rem;padding-bottom:4rem;}
h1{font-size:2.25rem!important;line-height:1.2!important;}
p,li{font-size:17px;line-height:1.65;}
.hero{background:#173e32;color:#fff;padding:28px 30px;border-radius:14px;margin-bottom:24px;border-left:6px solid #d7c17b;}
.hero h2{color:#fff!important;margin:0!important;font-size:28px!important;}
.hero p{color:#dce9e1;margin-bottom:0;}
[data-testid="stCaptionContainer"]{opacity:1!important;}
[data-testid="stCaptionContainer"] p{color:#44544a!important;}
[data-testid="stButton"] button{min-height:44px;}

[data-testid="stMetric"]{padding:14px;background:#eef4ef;border-radius:8px;}
[data-testid="stCaptionContainer"] p{color:#526057;}
</style>''',unsafe_allow_html=True)
@st.cache_resource
def load():
 model=XGBClassifier();model.load_model(P/'model.ubj')
 meta=json.loads((P/'metadata.json').read_text());data=pd.read_csv(P/'examples.csv')
 probs=model.predict_proba(data[meta['features']])
 data['prediction']=[meta['classes'][i] for i in probs.argmax(axis=1)]
 data['correct']=data.prediction==data.tactic_label
 return model,meta,data,probs
m,meta,data,all_probs=load();features=meta['features'];classes=meta['classes']
DEFS={
'build_up':'A controlled attack that usually begins farther from the opponent’s goal.',
'creation':'Developing an attacking opportunity.',
'direct_attack':'Moving the ball forward quickly.',
'fast_transition':'Attacking soon after gaining possession.',
'finishing':'The final part of an attack near the opponent’s goal.'}
FIELDS={
'ball_x_start':('Ball starting x position','m','Position along the pitch at the start of the phase. Zero is midfield. These are the saved tracking coordinates; the app does not infer attacking direction.'),
'ball_progress_m':('Signed ball progress','m','Change in the ball’s x coordinate over the phase. A negative value means movement toward decreasing x, not automatically a backward attack.'),
'duration':('Phase duration','s','How long this section of play lasted.'),
'progress_speed_mps':('Signed progress speed','m/s','Signed x progress divided by duration. This is not the ball’s total running speed.'),
'attack_width_mean':('Attacking team width','m','Average side-to-side spread of the attacking team.'),
'attack_depth_mean':('Attacking team depth','m','Average front-to-back spread of the attacking team.'),
'attack_spacing_mean':('Attacking player spacing','m','Phase-average spacing measurement for attacking players.'),
'defense_width_mean':('Defending team width','m','Average side-to-side spread of the defending team.'),
'defense_depth_mean':('Defending team depth','m','Average front-to-back spread of the defending team.'),
'defense_spacing_mean':('Defending player spacing','m','Phase-average spacing measurement for defending players.'),
'defenders_within_5m_mean':('Defenders within 5 m','players','Average number of defenders within 5 meters of the ball.'),
'defenders_within_10m_mean':('Defenders within 10 m','players','Average number of defenders within 10 meters of the ball.'),
'attackers_ahead_of_ball_mean':('Attackers ahead of the ball','players','Average count across the phase; fractional values are possible.'),
'attackers_in_wide_zones_mean':('Attackers in wide zones','players','Average count of attackers in the dataset’s wide zones.'),
'ball_wide_share':('Ball wide share','share','Fraction of sampled observations with the ball in the dataset’s wide zones.'),
'frames_sampled':('Sampled frames','frames','Number of tracking observations used to summarize the phase.'),
'match_minute':('Match minute','min','Time in the match associated with this phase.'),
'period':('Match period','period','Period identifier in the source data.')}
def chart(fig,key):
 fig=style(fig);fig.update_layout(height=430,margin=dict(l=20,r=25,t=65,b=70))
 st.plotly_chart(fig,width='stretch',key=key,config={'displaylogo':False,'scrollZoom':False})
def values_table(row):
 return pd.DataFrame([{'Measurement':FIELDS[f][0],'Value':round(float(row[f]),3),'Unit':FIELDS[f][1],'Meaning':FIELDS[f][2]} for f in features])

st.caption('Jack Chiang · Research companion')
st.title('Machine Learning-Based Classification of Soccer Attacking Phases')
st.markdown('**Using Spatiotemporal Tracking Features**')
page=st.radio('Explore the research',['1 · Understand the study','2 · Inspect real phases','3 · Compare the results'],horizontal=True,label_visibility='collapsed')
st.markdown('<div class="hero"><h2>'+({'1':'Read the game through data','2':'Inside one attacking phase','3':'Put the models to the test'}[page[0]])+'</h2><p>'+({'1':'Discover the five labels, then investigate the real examples behind them.','2':'Choose a case. Inspect the evidence. Explore how the model responds.','3':'Compare performance and see where classification becomes difficult.'}[page[0]])+'</p></div>',unsafe_allow_html=True)
if page.startswith('1'):
 st.header('Can tracking measurements identify an attacking phase?')
 st.write('Coaches and analysts often identify tactics by watching matches. My paper tests whether measurements of the players and ball can help classify short sections of an attack into five categories.')
 a,b,c=st.columns(3);a.metric('Matches','10');b.metric('Attacking phases','2,955');c.metric('Measurements per phase','18')
 st.caption('Source: paper, Dataset section. SkillCorner open data from the Australian A-League.')
 st.subheader('What the model receives—and what it returns')
 st.markdown('**Tracking observations → 18 phase measurements → classifier → one of five labels**')
 st.write('Each row summarizes one attacking phase: ball movement, team shape, player spacing, nearby defenders, and timing. The classifier learns to reproduce the phase labels supplied by SkillCorner’s AI-based system.')
 st.info('This app works with phase summaries. It does not contain match video or frame-by-frame player positions. Use “Inspect real phases” to examine the measurements and the model’s prediction together.')
 st.subheader('The five labels')
 tactic=st.selectbox('Choose a label to learn about',classes,format_func=lambda x:NAMES[x])
 st.markdown('### '+NAMES[tactic]);st.write(DEFS[tactic])
 counts={'build_up':598,'creation':1226,'direct_attack':337,'fast_transition':132,'finishing':662}
 st.write(f"The full study contains **{counts[tactic]:,} {NAMES[tactic].lower()} phases** ({counts[tactic]/2955:.1%} of the dataset). These descriptions are the paper’s explanations, not fixed numerical rules for assigning labels.")
 frame=pd.DataFrame({'Tactic':[NAMES[x] for x in classes],'Phases':[counts[x] for x in classes]})
 fig=px.bar(frame,x='Phases',y='Tactic',orientation='h',text='Phases',title='How many examples of each tactic?',color_discrete_sequence=['#285943'])
 fig.update_layout(yaxis=dict(autorange='reversed'));chart(fig,'counts')
 st.caption('All 2,955 phases. Uneven class sizes matter: a model can score well overall while missing a large share of a rare tactic.')
 st.subheader('How the study was tested')
 a,b=st.columns(2)
 a.markdown('**Train: 8 matches · 2,388 phases**\n\nThe models learn relationships between measurements and labels.')
 b.markdown('**Test: 2 different matches · 567 phases**\n\nThe models are evaluated on matches they did not see during training.')
 st.write('The original models used 14 measurements. The enhanced models added match minute, period, phase duration, and progress speed. Models were compared using the same match-based split.')
 st.subheader('Main finding')
 st.write('Enhanced XGBoost reached **91.9% accuracy** in the paper—about 92 correct labels per 100 test phases. Enhanced Random Forest had the highest **macro F1 (0.873)**, which gives the five tactics equal weight. Fast transitions remained the hardest category.')
 st.caption('Source: paper, Introduction, Dataset, Methodology, and Results and Discussion. Select “Compare the results” for the full comparison and limitations.')

elif page.startswith('2'):
 st.header('Inspect a real attacking phase')
 st.write('Choose a supplied label, then examine an actual phase from the two test matches. Compare its measurements with the model’s prediction. Nothing here is a reconstructed match replay.')
 st.caption('The explorer uses the retrained app model. Its results are separate from the original paper’s results.')
 a,b=st.columns(2)
 label=a.selectbox('1. Supplied tactic label',classes,format_func=lambda x:NAMES[x])
 mode=b.radio('Show examples',['All','Model matches label','Model differs from label'])
 pool=data[data.tactic_label==label]
 if mode!='All': pool=pool[pool.correct==(mode=='Model matches label')]
 st.write('**'+NAMES[label]+':** '+DEFS[label])
 st.caption(f'{len(pool)} test phases in this selection. A “supplied label” is SkillCorner’s reference classification, not an independent coach assessment.')
 if pool.empty:
  st.info('No phases match this filter. Select “All” to inspect this tactic.')
 else:
  options=pool.index.tolist()
  selection_key=f'phase_choice_{label}_{mode}'
  if st.session_state.get(selection_key) not in options: st.session_state[selection_key]=options[0]
  position=options.index(st.session_state[selection_key])
  nav1,nav2,nav3=st.columns([1,2,1])
  if nav1.button('← Previous',disabled=position==0):
   st.session_state[selection_key]=options[position-1]
   st.rerun()
  nav2.caption(f'Browse {len(options)} real test phases · use the buttons or pick an ID below')
  if nav3.button('Next →',disabled=position==len(options)-1):
   st.session_state[selection_key]=options[position+1]
   st.rerun()
  idx=st.selectbox('2. Choose a phase',options,key=selection_key,format_func=lambda i:f"Match {int(data.loc[i,'match_id'])} · phase {int(data.loc[i,'phase_index'])} · minute {data.loc[i,'match_minute']:.1f}")
  row=data.loc[idx];prob=all_probs[idx];winner=row.prediction
  a,b=st.columns(2);a.metric('Supplied label',NAMES[label]);b.metric('App prediction',NAMES[winner])
  st.write('The app model **'+('matches' if row.correct else 'differs from')+'** the supplied label for this phase.')
  rank=np.argsort(prob)[::-1]
  st.info(f"The model gives {NAMES[classes[rank[0]]]} {prob[rank[0]]:.1%} probability and {NAMES[classes[rank[1]]]} {prob[rank[1]]:.1%}. "+('The top choice matches the supplied label.' if row.correct else 'This is a useful error case: inspect the overlapping measurements below.'))
  activity=st.radio('What would you like to investigate?',['Read this phase','Change an input'],horizontal=True)
  st.subheader('Measurements behind this example')
  a,b,c=st.columns(3);a.metric('Duration',f'{row.duration:.1f} s');b.metric('Signed x progress',f'{row.ball_progress_m:+.1f} m');c.metric('Attacking width',f'{row.attack_width_mean:.1f} m')
  st.caption('Progress is displacement along the stored x-axis. Width is the team’s average side-to-side spread. These summaries do not show the order of movement.')
  choices=['duration','ball_x_start','ball_progress_m','attack_width_mean','attackers_ahead_of_ball_mean']
  if activity=='Read this phase':
   scores=pd.DataFrame({'Tactic':[NAMES[c] for c in classes],'Model probability':prob})
   fig=px.bar(scores,x='Model probability',y='Tactic',orientation='h',text='Model probability',range_x=[0,1],title='How the app model scores this phase',color_discrete_sequence=['#285943'])
   fig.update_traces(texttemplate='%{x:.1%}');fig.update_xaxes(tickformat='.0%');chart(fig,'probabilities')
   st.caption('The highest score determines the predicted label. These are model probabilities, not guarantees or an explanation of why a prediction was made.')
   st.subheader('How does this phase compare with other phases?')
   choices=['duration','ball_x_start','ball_progress_m','attack_width_mean','attackers_ahead_of_ball_mean']
   f=st.selectbox('Compare one measurement',choices,format_func=lambda x:FIELDS[x][0])
   title,unit,meaning=FIELDS[f];st.write(meaning)
   view=data.assign(Tactic=data.tactic_label.map(NAMES))
   fig=px.box(view,x=f,y='Tactic',orientation='h',category_orders={'Tactic':list(NAMES.values())},labels={f:f'{title} ({unit})'},title=f'{title} across all 567 test phases',color_discrete_sequence=['#819b8a'])
   fig.add_vline(x=float(row[f]),line_color='#a4471a',line_width=3,line_dash='dash',annotation_text=f'Selected phase: {row[f]:.1f} {unit}',annotation_position='top')
   chart(fig,'distribution')
   st.caption('The line inside each box is the median; the box contains the middle half of the values. The orange dashed line marks your selected phase. Overlapping boxes show why one measurement alone cannot define a tactic.')
   with st.expander('See all 18 measurements and their meanings'):
    st.dataframe(values_table(row),hide_index=True,width='stretch')
  if activity=='Change an input':
   st.subheader('See the model respond')
   st.write('This is a sensitivity check: how does the model respond if one measurement changes? It is not a simulation of how players would react. All other inputs stay fixed, except progress speed is recalculated when duration or progress changes.')
   field=st.selectbox('Measurement to change',choices,format_func=lambda x:FIELDS[x][0],key='experiment_field',index=1)
   st.write(FIELDS[field][2])
   low,high=float(data[field].min()),float(data[field].max())
   experiment_key=f'experiment_{idx}_{field}'
   if st.button('Restore original value'): st.session_state.pop(experiment_key,None)
   val=st.slider(f'{FIELDS[field][0]} ({FIELDS[field][1]})',low,high,float(row[field]),key=f'experiment_{idx}_{field}')
   altered=row[features].astype(float).copy();altered[field]=val
   if field in ['duration','ball_progress_m']: altered['progress_speed_mps']=altered.ball_progress_m/altered.duration
   new=m.predict_proba(pd.DataFrame([altered],columns=features))[0];newlabel=classes[int(new.argmax())]
   st.caption(f'Input change: {row[field]:.2f} → {val:.2f} {FIELDS[field][1]}')
   if np.isclose(val,float(row[field])):
    st.info('Move the slider to test a different value. The current value is the original measurement.')
   elif newlabel==winner:
    st.info(f'The input changed, but the top prediction is still {NAMES[winner]}. A tree model can keep the same answer across a range of values. The curve below shows its response across the full slider range.')
   else:
    st.success(f'The top prediction changed from {NAMES[winner]} to {NAMES[newlabel]}.')
   st.write(f'**Original prediction: {NAMES[winner]} → changed-input prediction: {NAMES[newlabel]}**')
   sweep=np.linspace(low,high,80)
   sweep_x=pd.DataFrame([row[features].astype(float).to_dict() for _ in sweep])
   sweep_x[field]=sweep
   if field in ['duration','ball_progress_m']: sweep_x['progress_speed_mps']=sweep_x.ball_progress_m/sweep_x.duration
   sweep_p=m.predict_proba(sweep_x[features])
   response=pd.DataFrame(sweep_p,columns=[NAMES[c] for c in classes]);response[FIELDS[field][0]]=sweep
   fig=px.line(response.melt(id_vars=FIELDS[field][0],var_name='Tactic',value_name='Probability'),x=FIELDS[field][0],y='Probability',color='Tactic',color_discrete_map=COLORS,range_y=[0,1],title='Where does the model change its mind?',labels={FIELDS[field][0]:f'{FIELDS[field][0]} ({FIELDS[field][1]})'})
   fig.update_traces(line_shape='hv');fig.update_yaxes(tickformat='.0%')
   fig.add_vline(x=val,line_dash='dash',line_color='#17251c',annotation_text='Your value',annotation_position='top')
   chart(fig,'response_curve')
   st.caption('Each line is one tactic’s probability at 80 sampled input values. The dashed line is your selection. Hover for scores; click a legend label to hide a tactic. These are hypothetical model responses, not observed match outcomes.')
   comp=pd.DataFrame({'Tactic':[NAMES[c] for c in classes],'Original probability':prob,'Changed probability':new})
   fig=px.bar(comp.melt(id_vars='Tactic',var_name='Version',value_name='Probability'),x='Tactic',y='Probability',color='Version',barmode='group',range_y=[0,1],title='Original and changed-input scores',color_discrete_map={'Original probability':'#a3b6aa','Changed probability':'#285943'})
   fig.update_yaxes(tickformat='.0%');chart(fig,'experiment')
   st.caption('Slider limits are the observed test-data range. Combining a new value with the other unchanged measurements may still describe an unrealistic phase. These predictions are not results reported in the paper.')

else:
 st.header('What did the paper find?')
 st.write('The results below are transcribed from the supplied paper. Accuracy and macro F1 answer different questions, so the model with the highest accuracy is not necessarily strongest across all five tactics.')
 results=pd.DataFrame({'Model':['Original Random Forest','Enhanced Random Forest','Original XGBoost','Enhanced XGBoost','SVC','MLP','KNN'],'Accuracy':[.912,.912,.914,.919,.879,.858,.756],'Macro F1':[.863,.873,.870,.864,.825,.790,.649]})
 metric=st.radio('Compare models by',['Accuracy','Macro F1'],horizontal=True)
 st.write('**Accuracy:** the fraction of all test phases labeled correctly.' if metric=='Accuracy' else '**Macro F1:** precision and recall are combined for each tactic, then the five scores are averaged equally. This gives rare tactics as much weight as common ones.')
 fig=px.bar(results.sort_values(metric),x=metric,y='Model',orientation='h',text=metric,range_x=[0,1],title=f'Paper results · {metric}',color_discrete_sequence=['#285943'])
 fig.update_traces(texttemplate='%{x:.1%}' if metric=='Accuracy' else '%{x:.3f}')
 if metric=='Accuracy': fig.update_xaxes(tickformat='.0%')
 chart(fig,'model_comparison')
 st.caption('Source: Results and Discussion. Original = 14 inputs; enhanced = 18 inputs. Every model uses the same match-based train/test split.')
 st.subheader('Why overall accuracy is not the whole story')
 f1=pd.DataFrame({'Tactic':list(NAMES.values()),'F1':[.98,.92,.91,.59,.91]})
 fig=px.bar(f1,x='F1',y='Tactic',orientation='h',text='F1',range_x=[0,1],title='Paper: Enhanced XGBoost score for each tactic',color_discrete_sequence=['#426b99']);chart(fig,'f1')
 st.write('The paper reports **13 of 25 fast transitions identified correctly**, versus all 117 build-up phases. Fast transitions were often confused with creation or finishing. They had fewer training examples, and phase averages lose the sequence of events that distinguishes a transition from an established attack.')
 st.caption('F1 is not the percentage correctly classified: it combines precision and recall. Source: Results and Discussion, Figure 3 discussion.')
 st.subheader('Which measurements mattered?')
 importance=pd.DataFrame({'Measurement':['Ball starting position','Sampled frames','Ball progress','Attackers ahead of the ball'],'Importance':[.1961,.1187,.1124,.1018]})
 chart(px.bar(importance,x='Importance',y='Measurement',orientation='h',text='Importance',title='Four feature-importance values reported in the paper',color_discrete_sequence=['#8c6d35']),'importance')
 st.write('These scores describe how the trained XGBoost trees use the inputs. They do not show cause and effect or explain one specific prediction. The paper also reports that shuffling ball starting position and progress had the largest effects on model performance.')
 with st.expander('Feature removal, validation, and tuning'):
  st.write('Removing weaker measurements produced only small changes. The paper does not establish that fewer features are clearly better.')
  ab=pd.DataFrame({'Inputs retained':[14,12,11,9],'Accuracy':[.912,.910,.910,.912],'Macro F1':[.863,.858,.861,.867]})
  st.dataframe(ab,hide_index=True,width='stretch')
  st.write('Across five match-based GroupKFold splits, average accuracy was around 91% and macro F1 around 0.85. The paper reports variation between matches; it does not provide exact fold scores in its prose.')
  st.write('Randomized Search tested 25 parameter combinations across five folds. The best validation macro F1 was 0.884. This is a validation score, not a new held-out test result.')
  st.caption('Best reported settings: 400 trees; depth 7; learning rate 0.03; row and column sampling 0.8; minimum child weight 5; regularization 1.')
 with st.expander('How the live app model differs from the paper'):
  st.write(f"The explorer uses a retrained XGBoost model with {meta['accuracy']:.1%} accuracy and {meta['macro_f1']:.3f} macro F1 on the 567 test phases. The paper’s Enhanced XGBoost scores are 91.9% and 0.864. Do not substitute app scores for the paper’s results.")
  chart(px.imshow(meta['confusion'],x=[NAMES[c] for c in classes],y=[NAMES[c] for c in classes],text_auto=True,labels={'x':'App prediction','y':'Supplied label','color':'Phases'},color_continuous_scale='Greens',title='Retrained app model · test-phase confusion matrix'),'confusion')
  st.caption('Read by row: supplied labels on the left, predictions along the bottom. Diagonal cells are matches. This matrix belongs to the app model, not the original paper.')
 st.subheader('What the results do—and do not—support')
 st.write('The study shows that phase summaries can reproduce SkillCorner’s labels with strong overall accuracy in these matches. The labels themselves were generated by SkillCorner’s system, rather than independently assigned by coaches for this study.')
 st.write('Only 10 matches from one league were studied. More teams and leagues are needed to assess generalization. Keeping the full sequence of player and ball movements could help distinguish rapidly changing tactics. The paper also proposes exploring score difference and the preceding tactic.')
 st.caption('Source: supplied paper, sections 5–7. Research acknowledgements: Omar Ramos Escoto, Inspirit AI, and SkillCorner.')
