from EEG_research_tool import App
from EEG_analysis_tool import analysis_frame

app = App()

analysis  = analysis_frame(app.root)
analysis.pack(fill = "both", expand = True)

app.run()



