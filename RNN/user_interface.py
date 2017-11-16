
from flexx import app, event, ui




class User_interface(ui.Widget):

    def init(self):
        with ui.HBox():
            self.b1 = ui.Button(flex=.25, text = "testing1" )
            self.b2 = ui.Button(flex=.25, text = "testing2" )
            self.b3 = ui.Button(flex=.25, text = "testing3" )
            self.b4 = ui.Button(flex=.25, text = "testing4" )
            self.b5 = ui.Button(flex=.25, text = "testing5",)



    @event.connect('b1.mouse_click')
    def _button1_click(self, *events):
        self.b1.text = "Updated text"
    @event.connect('b2.mouse_click')
    def _button2_click(self, *events):
        self.b2.text = "Upasdfassd text"
    @event.connect('b3.mouse_click')
    def _button3_click(self, *events):
        self.b3.text = "sfdatext"
    @event.connect('b4.mouse_click')
    def _button4_click(self, *events):
        self.b4.text = "more text"
    @event.connect('b5.mouse_click')
    def _button5_click(self, *events):
        self.b5.text = "fasdf text"


if __name__ == '__main__':
    m = app.launch(User_interface)
    app.run()