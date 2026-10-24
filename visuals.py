import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
NAMES={'build_up':'Build-up','creation':'Creation','direct_attack':'Direct attack','fast_transition':'Fast transition','finishing':'Finishing'}
COLORS=dict(zip(NAMES.values(),['#2563a0','#a67416','#775ca6','#bc5230','#277557']))
def style(fig):
 return fig.update_layout(template='plotly_white',font=dict(size=15,color='#202820'),paper_bgcolor='white',plot_bgcolor='white',margin=dict(l=15,r=20,t=45,b=25),legend=dict(orientation='h',y=-.25))
def pitch(row):
 fig=go.Figure()
 fig.add_shape(type='rect',x0=-55,y0=-35,x1=55,y1=35,line_color='white',fillcolor='#367558')
 fig.add_shape(type='line',x0=0,y0=-35,x1=0,y1=35,line_color='white')
 fig.add_shape(type='circle',x0=-9.15,y0=-9.15,x1=9.15,y1=9.15,line_color='white')
 for sign in [-1,1]:
  fig.add_shape(type='rect',x0=sign*55,y0=-20,x1=sign*38.5,y1=20,line_color='white')
 x=float(row['ball_x_start']);end=x+float(row['ball_progress_m'])
 fig.add_trace(go.Scatter(x=[x,end],y=[0,0],mode='lines+markers+text',text=['Start','End'],textposition=['top center','bottom center'],line=dict(color='#ffe18b',width=5),marker=dict(size=13,color='#ffe18b'),name='Ball x position',hovertemplate='x = %{x:.1f} m<extra></extra>'))
 fig.update_layout(height=310,showlegend=False,margin=dict(l=10,r=10,t=10,b=10),paper_bgcolor='white',plot_bgcolor='#367558',font=dict(color='#17251c',size=15))
 fig.update_xaxes(range=[min(-58,x-5,end-5),max(58,x+5,end+5)],visible=False);fig.update_yaxes(range=[-38,38],visible=False,scaleanchor='x')
 return fig
