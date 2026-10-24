import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
NAMES={'build_up':'Build-up','creation':'Creation','direct_attack':'Direct attack','fast_transition':'Fast transition','finishing':'Finishing'}
COLORS=dict(zip(NAMES.values(),['#2563a0','#a67416','#775ca6','#bc5230','#277557']))
def style(fig):
 return fig.update_layout(template='plotly_white',font=dict(size=15,color='#202820'),paper_bgcolor='white',plot_bgcolor='white',margin=dict(l=15,r=20,t=45,b=25),legend=dict(orientation='h',y=-.25))
