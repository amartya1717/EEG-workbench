import tkinter as tk

class App:

    def __init__(self):
        
        self.root = tk.Tk()
        # self.root.geometry('1920x1080')
        self.root.attributes("-fullscreen",True)
        self.root.bind("<Escape>",self.exit_fullscreen)
        


    def exit_fullscreen(self,event = None):
        self.root.attributes("-fullscreen", False)
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{screen_width}x{screen_height}+0+0")
        self.root.resizable(True, True)
        self.root.unbind("<Escape>")
        pass
   
    def run(self):
        self.root.mainloop()
        pass
