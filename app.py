from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from xgboost import XGBClassifier
from visuals import NAMES, COLORS, style

P=Path(__file__).parent
st.set_page_config(page_title='Soccer Attacking Phases | Jack Chiang',layout='wide')
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
'match_minute':('Minutes elapsed in half','min','Elapsed time within the recorded half, restarting at zero for the second half. This is not the full-match clock.'),
'period':('Match half','half','1 = first half; 2 = second half.')}
def chart(fig,key):
 fig=style(fig);fig.update_layout(height=430,margin=dict(l=20,r=25,t=65,b=70))
 st.plotly_chart(fig,width='stretch',key=key,config={'displaylogo':False,'scrollZoom':False})
def phase_description(row):
 period=int(row['period'])
 seconds=int(round(float(row['match_minute'])*60))
 return f"Match {int(row['match_id'])} · phase {int(row['phase_index'])} · half {period}, {seconds//60}:{seconds%60:02d} elapsed"

def values_table(row):
 return pd.DataFrame([{'Measurement':FIELDS[f][0],'Value':round(float(row[f]),3),'Unit':FIELDS[f][1],'Meaning':FIELDS[f][2]} for f in features])

st.title('Machine Learning-Based Classification of Soccer Attacking Phases')
st.markdown('**Using Spatiotemporal Tracking Features**')
page=st.radio('Explore the research',['1 · Understand the study','2 · Inspect real phases','3 · Compare the results'],horizontal=True,label_visibility='collapsed')
st.markdown('<div class="hero"><h2>'+({'1':'Read the game through data','2':'Inside one attacking phase','3':'Compare the evidence'}[page[0]])+'</h2><p>'+({'1':'Discover the five labels, then investigate the real examples behind them.','2':'Choose a case. Inspect the evidence. Explore how the model responds.','3':'Compare performance and see where classification becomes difficult.'}[page[0]])+'</p></div>',unsafe_allow_html=True)
if page.startswith('1'):
 st.header('What is an attacking phase?')
 st.write('An attack can change character: a team may build from the back, work for an opening, or rush forward after winning the ball. A phase is a short section of play given one of these labels. The five labels are categories, not five steps every attack must follow.')
 st.subheader('Start with a soccer example')
 tactic=st.selectbox('Choose a label to learn about',classes,format_func=lambda x:NAMES[x])
 stories={
 'build_up':('A team has possession deep in its own half and moves the ball out from the back.','Build-up describes establishing the attack.',[(12,24),(25,38),(38,29)]),
 'creation':('A team already has possession and moves the ball around to look for an opening.','Creation describes developing a chance.',[(55,20),(67,40),(77,28)]),
 'direct_attack':('A team sends the ball a long way forward quickly instead of using a patient series of passes.','Direct attack describes a quick route forward; it does not by itself tell you when possession was won.',[(25,25),(73,35),(88,28)]),
 'fast_transition':('A team wins possession and immediately attacks before the opponent can reorganize.','The key context is the change of possession. A fast forward attack is not automatically a fast transition.',[(40,37),(61,29),(82,38)]),
 'finishing':('An attack reaches its final action near the opponent’s goal, such as preparing a shot.','Finishing describes the end of an attacking move, not necessarily a goal.',[(77,42),(87,30),(98,30)])}
 story,lesson,points=stories[tactic]
 st.markdown('### '+NAMES[tactic]);st.write(story)
 fig=go.Figure()
 fig.add_shape(type='rect',x0=0,y0=0,x1=105,y1=68,line_color='white',fillcolor='#285943')
 fig.add_shape(type='line',x0=52.5,y0=0,x1=52.5,y1=68,line_color='white')
 fig.add_shape(type='circle',x0=43.35,y0=24.85,x1=61.65,y1=43.15,line_color='white')
 for x0,x1 in [(0,16.5),(88.5,105)]: fig.add_shape(type='rect',x0=x0,y0=13.84,x1=x1,y1=54.16,line_color='white')
 for (x,y),(nx,ny) in zip(points,points[1:]): fig.add_annotation(x=nx,y=ny,ax=x,ay=y,xref='x',yref='y',axref='x',ayref='y',showarrow=True,arrowhead=3,arrowwidth=4,arrowcolor='#f4cd63')
 fig.add_trace(go.Scatter(x=[v[0] for v in points],y=[v[1] for v in points],mode='markers+text',marker=dict(size=24,color='#f4cd63'),text=['1','2','3'],textposition='middle center',textfont=dict(color='#173e32'),hoverinfo='skip'))
 fig.update_layout(height=340,margin=dict(l=10,r=10,t=35,b=10),title='Illustration · attack moves left to right',showlegend=False,paper_bgcolor='white',plot_bgcolor='white',xaxis=dict(range=[-2,107],visible=False,fixedrange=True),yaxis=dict(range=[-2,70],visible=False,scaleanchor='x',scaleratio=1,fixedrange=True))
 st.plotly_chart(fig,width='stretch',config={'displayModeBar':False},key='concept_pitch')
 st.caption('Illustration only: the numbered points show an imagined ball route. This is not match footage, recorded tracking, or a model prediction.')
 st.write('**What to notice:** '+lesson)
 st.subheader('What does the paper ask?')
 st.write('Can a computer identify these categories from measurements of the ball and players? Each example becomes a row of 18 numbers, including duration, team width, and ball position. The model chooses one label, which is compared with the label already supplied by SkillCorner.')
 a,b,c=st.columns(3);a.metric('Matches studied','10');b.metric('Sections of play','2,955');c.metric('Inputs per example','18')
 st.caption('Australian A-League · SkillCorner open data. The saved examples contain measurements, not videos or full player paths.')
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
 st.write('Start with one example the model gets right. Then choose a close call or a mistake to see why identifying a phase can be difficult. “Browse freely” opens the full set of recorded examples.')
 st.caption('The explorer uses the retrained app model. Its results are separate from the original paper’s results.')
 guide=st.radio('Start with a case',['Correct classification','Close call','Fast-transition mistake','Browse freely'],horizontal=True)
 guided_index=None
 if guide!='Browse freely':
  margin=np.sort(all_probs,axis=1)[:,-1]-np.sort(all_probs,axis=1)[:,-2]
  if guide=='Correct classification':
   candidates=np.flatnonzero(data.correct.to_numpy())
   typicality=(data.groupby('tactic_label')[features].rank(pct=True)-0.5).abs().mean(axis=1).to_numpy()
   guided_index=int(candidates[np.argmin(typicality[candidates])])
   st.write('A correct example whose measurements are relatively typical within its supplied tactic. Check which label leads, then compare its measurements with the other tactics.')
  elif guide=='Close call':
   guided_index=int(np.argmin(margin))
   st.write('The two leading scores are closest in this example. A close call can still be correct; inspect both scores before drawing conclusions.')
  else:
   candidates=np.flatnonzero(((data.tactic_label=='fast_transition') & ~data.correct).to_numpy())
   if len(candidates): guided_index=int(candidates[np.argmin(margin[candidates])])
   st.write('A phase labeled fast transition that the app predicts differently. This illustrates the challenge discussed in the paper; it does not reproduce the paper’s original model output.')
  if guided_index is not None:
   st.session_state['label_filter']=data.loc[guided_index,'tactic_label']
   st.session_state['outcome_filter']='All'
   st.session_state[f"phase_choice_{data.loc[guided_index,'tactic_label']}_All"]=guided_index

 if guide!='Browse freely':
  idx=guided_index
  label=data.loc[idx,'tactic_label']
  pool=data.loc[[idx]]
 else:
  a,b=st.columns(2)
  label=a.selectbox('1. Supplied tactic label',classes,format_func=lambda x:NAMES[x],key='label_filter',disabled=guide!='Browse freely')
  mode=b.radio('Show examples',['All','Model matches label','Model differs from label'],key='outcome_filter',disabled=guide!='Browse freely')
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
   if nav1.button('← Previous',disabled=position==0 or guide!='Browse freely'):
    st.session_state[selection_key]=options[position-1]
    st.rerun()
   nav2.caption(f'Phase {position+1} of {len(options)} in this selection' if guide=='Browse freely' else 'Guided case · choose Browse freely to change phases')
   if nav3.button('Next →',disabled=position==len(options)-1 or guide!='Browse freely'):
    st.session_state[selection_key]=options[position+1]
    st.rerun()
   idx=st.selectbox('2. Choose a phase',options,key=selection_key,format_func=lambda i:phase_description(data.loc[i]),disabled=guide!='Browse freely')
 if not pool.empty:
  row=data.loc[idx];prob=all_probs[idx];winner=row.prediction
  if guide!='Browse freely':
   st.write('**'+NAMES[label]+':** '+DEFS[label])
   st.caption(phase_description(row))
  st.caption('The supplied label is SkillCorner’s classification of this recorded phase. The app model tries to reproduce it.')
  a,b=st.columns(2);a.metric('Supplied label',NAMES[label]);b.metric('Original app prediction',NAMES[winner])
  rank=np.argsort(prob)[::-1]
  st.info(f"The model gives {NAMES[classes[rank[0]]]} {prob[rank[0]]:.1%} probability and {NAMES[classes[rank[1]]]} {prob[rank[1]]:.1%}. "+('The top choice matches the supplied label.' if row.correct else 'The top choice differs from the supplied label.'))
  activity=st.radio('What would you like to investigate?',['Read this phase','Change an input'],horizontal=True)
  st.write('**What was recorded?** This section of play lasted '+f'{row.duration:.1f} seconds'+'. The attacking team was spread '+f'{row.attack_width_mean:.1f} meters'+' across the pitch on average.')
  st.subheader('Original recorded measurements')
  a,b,c=st.columns(3);a.metric('Duration',f'{row.duration:.1f} s');b.metric('Signed x progress',f'{row.ball_progress_m:+.1f} m');c.metric('Attacking width',f'{row.attack_width_mean:.1f} m')
  st.caption('Progress is displacement along the stored x-axis. Width is the team’s average side-to-side spread. These summaries do not show the order of movement.')
  choices=['duration','ball_x_start','ball_progress_m','attack_width_mean','attackers_ahead_of_ball_mean']
  if activity=='Read this phase':
   scores=pd.DataFrame({'Tactic':[NAMES[c] for c in classes],'Model probability':prob})
   fig=px.bar(scores,x='Model probability',y='Tactic',orientation='h',text='Model probability',range_x=[0,1],title='How the app model scores this phase',color_discrete_sequence=['#285943'])
   fig.update_traces(texttemplate='%{x:.1%}');fig.update_xaxes(tickformat='.0%');chart(fig,'probabilities')
   st.caption('The highest score determines the predicted label. These are model probabilities, not guarantees or an explanation of why a prediction was made.')
   st.subheader('How does this phase compare with other phases?')
   f=st.selectbox('Compare one measurement',choices,format_func=lambda x:FIELDS[x][0])
   title,unit,meaning=FIELDS[f];st.write(meaning)
   same_label=data.loc[data.tactic_label==label,f]
   middle=float(same_label.median())
   below=int((same_label<float(row[f])).sum())
   st.write(f'**Your phase: {row[f]:.1f} {unit}.** The median for {NAMES[label].lower()} is {middle:.1f} {unit}; {below} of {len(same_label)} recorded {NAMES[label].lower()} phases have a lower value.')
   view=data.assign(Tactic=data.tactic_label.map(NAMES))
   fig=px.box(view,x=f,y='Tactic',orientation='h',category_orders={'Tactic':list(NAMES.values())},labels={f:f'{title} ({unit})'},title=f'{title} across all 567 test phases',color_discrete_sequence=['#819b8a'])
   fig.add_vline(x=float(row[f]),line_color='#a4471a',line_width=3,line_dash='dash',annotation_text=f'Selected phase: {row[f]:.1f} {unit}',annotation_position='top')
   chart(fig,'distribution')
   st.caption('The line inside each box is the median; the box contains the middle half of the values. The orange dashed line marks your selected phase. Overlapping boxes show why one measurement alone cannot define a tactic.')
   with st.expander('See all 18 measurements and their meanings'):
    st.dataframe(values_table(row),hide_index=True,width='stretch')
   with st.expander('Source-data checks and interpretation'):
    st.write('The supplied summaries include signed progress-speed values from −252.5 to +190.0 m/s and an average attackers-ahead count above 10 in one phase. The speed extremes require an extraction check before physical interpretation; the player-count definition also needs verification. The app preserves the supplied values so predictions remain reproducible; the underlying feature extraction needs review before interpreting them physically.')
    st.caption('A half-relative clock is shown above. No video, player identities, or frame sequence is available in these saved phase summaries to resolve the flagged measurements.')
  if activity=='Change an input':
   st.subheader('See the model respond')
   st.write('Change a spatial measurement to test the classifier. The other 17 inputs stay fixed. This is a hypothetical input test, not a reconstruction of how an attack would unfold.')
   st.caption('Duration and progress are read-only here because changing them also affects speed and may conflict with the sampled tracking observations.')
   field=st.selectbox('Measurement to change',['ball_x_start','attack_width_mean','attackers_ahead_of_ball_mean'],format_func=lambda x:FIELDS[x][0],key='experiment_field')
   st.write(FIELDS[field][2])
   low,high=float(data[field].min()),float(data[field].max())
   experiment_key=f'experiment_{idx}_{field}'
   if st.button('Restore original value'): st.session_state.pop(experiment_key,None)
   val=st.slider(f'{FIELDS[field][0]} ({FIELDS[field][1]})',low,high,float(row[field]),key=f'experiment_{idx}_{field}')
   altered=row[features].astype(float).copy();altered[field]=val
   new=m.predict_proba(pd.DataFrame([altered],columns=features))[0];newlabel=classes[int(new.argmax())]
   st.caption(f'Input change: {row[field]:.2f} → {val:.2f} {FIELDS[field][1]}')
   before,after=st.columns(2)
   before.metric('Prediction with original inputs',NAMES[winner])
   after.metric('Prediction with your input',NAMES[newlabel])
   st.caption(f'The changed model assigns {float(new.max()):.1%} to its top choice. The supplied reference label remains {NAMES[label]}; no ground-truth label exists for the hypothetical input.')
   if np.isclose(val,float(row[field])):
    st.info('Move the slider to test a different value. The current value is the original measurement.')
   elif newlabel==winner:
    st.info(f'The input changed, but the top prediction is still {NAMES[winner]}. A tree model can keep the same answer across a range of values. The curve below shows its response across the full slider range.')
   else:
    st.success(f'The top prediction changed from {NAMES[winner]} to {NAMES[newlabel]}.')
   sweep=np.unique(np.append(np.linspace(low,high,80),[val,float(row[field])]))
   sweep_x=pd.DataFrame([row[features].astype(float).to_dict() for _ in sweep])
   sweep_x[field]=sweep
   sweep_p=m.predict_proba(sweep_x[features])
   response=pd.DataFrame(sweep_p,columns=[NAMES[c] for c in classes]);response[FIELDS[field][0]]=sweep
   fig=px.line(response.melt(id_vars=FIELDS[field][0],var_name='Tactic',value_name='Probability'),x=FIELDS[field][0],y='Probability',color='Tactic',color_discrete_map=COLORS,range_y=[0,1],title='Model response at sampled input values',labels={FIELDS[field][0]:f'{FIELDS[field][0]} ({FIELDS[field][1]})'})
   fig.update_traces(mode='lines+markers',marker_size=4);fig.update_yaxes(tickformat='.0%')
   fig.add_vline(x=val,line_dash='dash',line_color='#17251c',annotation_text='Your value',annotation_position='top')
   chart(fig,'response_curve')
   st.caption('Dots are model evaluations at 80 evenly spaced values plus the original and selected values. Connecting lines are visual guides; transition locations between samples are approximate. The dashed line is your exact selection. Hover for scores; click a legend label to hide a tactic. These are hypothetical model responses, not observed match outcomes.')
   comp=pd.DataFrame({'Tactic':[NAMES[c] for c in classes],'Original score':[f'{v:.2%}' for v in prob],'Changed score':[f'{v:.2%}' for v in new],'Change (percentage points)':[f'{100*v:+.2f}' for v in new-prob]})
   st.dataframe(comp,hide_index=True,width='stretch')
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
  st.write('The app uses 400 trees, maximum depth 5, and learning rate 0.05. The paper’s best randomized-search settings use depth 7 and learning rate 0.03. The deployed model is a separate retraining, not a reproduction of the tuned model.')
  st.write(f"The explorer uses a retrained XGBoost model with {meta['accuracy']:.1%} accuracy and {meta['macro_f1']:.3f} macro F1 on the 567 test phases. The paper’s Enhanced XGBoost scores are 91.9% and 0.864. Do not substitute app scores for the paper’s results.")
  chart(px.imshow(meta['confusion'],x=[NAMES[c] for c in classes],y=[NAMES[c] for c in classes],text_auto=True,labels={'x':'App prediction','y':'Supplied label','color':'Phases'},color_continuous_scale='Greens',title='Retrained app model · test-phase confusion matrix'),'confusion')
  st.caption('Read by row: supplied labels on the left, predictions along the bottom. Diagonal cells are matches. This matrix belongs to the app model, not the original paper.')
 st.subheader('What the results do—and do not—support')
 st.write('The study shows that phase summaries can reproduce SkillCorner’s labels with strong overall accuracy in these matches. The labels themselves were generated by SkillCorner’s system, rather than independently assigned by coaches for this study.')
 st.write('Only 10 matches from one league were studied. More teams and leagues are needed to assess generalization. Keeping the full sequence of player and ball movements could help distinguish rapidly changing tactics. The paper also proposes exploring score difference and the preceding tactic.')
 st.caption('Source: supplied paper, sections 5–7. Research acknowledgements: Omar Ramos Escoto, Inspirit AI, and SkillCorner.')
