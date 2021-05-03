import tkinter as tk
import argparse

import cv2
import numpy as np
from pandas import read_csv
from PIL import ImageTk, Image
from os import getcwd
from os.path import sep, isfile
from re import sub
from tkinter import  messagebox

from digitDetection.digit_predictor import DigitPredictor

NO_NUMBER = -1
STOP_TAGGING = -2
EXIT_POP_UP = -3
NO_DIGIT = -4
DIGIT_TAGGED = -5
MAX_DIGITS_IN_JERSEY = 2
WELCOME_MSG = "Welcome to the jersey number tagger\n" \
                               "You can exit by pressing the x in the top right corner or pressing escape.\n" \
                               "Next time you'll enter, you will continue from the picture you left off" \


def character_limit(entry_text):
    if len(entry_text.get()) > 0:
        entry_text.set(entry_text.get()[:1])
    entry_text.set(sub("[^0-9]", "", entry_text.get()))

class JerseyTaggerGUI(tk.Frame):
    def __init__(self, bb_image, master, digits_suggestions, digits_tagged, show_welcome_msg=False):
        assert len(digits_suggestions) > 0
        # main window widgets
        self.digits_suggestions = digits_suggestions
        self.digits_tagged = digits_tagged
        tk.Frame.__init__(self, master)
        self.master = master
        self.master.title("Manual NBA court tagger")
        self.master.setvar(name='result', value=False)
        self.master.setvar(name='digits_tagged', value=None)
        self.W, self.H = self.master.winfo_screenwidth() // 4, int(master.winfo_screenheight() / 2)

        self.newWindow = None
        tag_label_text = "Is the object in the picture \na basketball player in the game?\n"

        tag_question_label = tk.Label(self.master, text=tag_label_text)
        tag_question_label.config(font=("Arial", 14))
        tag_question_label.pack(pady=3)

        self.bb_image_label = tk.Label(master = self.master, borderwidth=2, relief="groove", text="bla")
        self.bb_image_pil = bb_image
        self.bb_image_tk = ImageTk.PhotoImage(bb_image.resize(self.get_resized_bb_image_dimensions(*bb_image.size)))
        self.bb_image_label.configure(image=self.bb_image_tk)
        self.bb_image_label.image = self.bb_image_tk
        self.bb_image_label.pack()

        self.buttons_frame = tk.LabelFrame(self.master, borderwidth=0)
        self.buttons_frame.pack(pady=10)
        self.yes_button = tk.Button(self.buttons_frame, text='Yes', width=5, pady=2,
                                    command=self.is_player_button_clicked)
        self.yes_button.pack(padx=20, pady=7, side='right')
        self.no_button = tk.Button(self.buttons_frame, text="No", command= self.is_not_player_button_clicked)
        self.no_button.pack(padx=20, side='left')

        master.protocol('WM_DELETE_WINDOW', self.close_main_gui)
        self.master.resizable(width=False, height=False)
        self.center(master)

        master.bind("<Escape>", lambda event: self.close_main_gui())
        if show_welcome_msg:
            tk.messagebox.showinfo("Players' jersey number tagger", WELCOME_MSG)

    def is_player_button_clicked(self):
        for digit_suggested in self.digits_suggestions:
            x1,y1,x2,y2 = digit_suggested[1]
            digit = digit_suggested[0]
            self.single_digit_gui(digit, (x1, y1, x2, y2))
            # waiting for the user's tag of the current digit
            self.master.wait_window(self.newWindow)
            if len(self.digits_tagged) == MAX_DIGITS_IN_JERSEY:
                break
        if len(self.digits_tagged) == 0:        # no digits were tagged
            self.master.setvar(name='result', value=NO_NUMBER)
        elif len(self.digits_tagged) > 0:
            print(str(len(self.digits_tagged)) + " digits were tagged")
            self.master.setvar(name='result', value=True)
        self.exit_gui()

    def is_not_player_button_clicked(self):
        self.master.setvar(name='result', value=NO_NUMBER)
        self.exit_gui()

    def get_resized_bb_image_dimensions(self, img_w, img_h):
        img_ratio = img_w / img_h
        img_to_win_ratio = 2 / 3
        return int((self.W * img_to_win_ratio) * img_ratio), int(self.H * img_to_win_ratio)

    def close_main_gui(self):
        res = tk.messagebox.askquestion('Exit Application', 'Do you really want to exit')
        if res == 'yes':
            self.master.setvar(name='result', value=STOP_TAGGING)
            self.exit_gui()

    def center(self, win, win_size=None):
        if not win_size:
            win_size = (self.W, self.H)
        width = win_size[0]
        frm_width = win.winfo_rootx() - win.winfo_x()
        win_width = width + 2 * frm_width
        height = win_size[1]
        titlebar_height = win.winfo_rooty() - win.winfo_y()
        win_height = height + titlebar_height + frm_width
        x = win.winfo_screenwidth() // 2 - win_width // 2
        y = win.winfo_screenheight() // 2 - win_height // 2 - 50
        win.geometry('{}x{}+{}+{}'.format(width, height, x, y))
        win.deiconify()

    def single_digit_gui(self, suggested_digit, digit_box_xyxy):
        self.master.withdraw()
        self.newWindow = tk.Toplevel(self.master)

        # convert to cv2 format and draw digit and it's bbox
        bb_image_cv = np.array(self.bb_image_pil)
        bb_image_with_digit_cv = self.draw_digit_box(bb_image_cv, box_xyxy=digit_box_xyxy)

        # convert back to tkinter format
        bb_image_pil_with_digit = Image.fromarray(cv2.resize(bb_image_with_digit_cv, (0, 0), fx=2, fy=2), 'RGB')
        bb_image_tk_with_digit = ImageTk.PhotoImage(image=bb_image_pil_with_digit)

        # add widgets to window
        game_image_label = tk.Label(self.newWindow, borderwidth=2, relief="groove")
        text = "Enter the digit inside the red&yellow bounding box\n" + \
               "if the digit is " + str(suggested_digit) + " you can press yes and continue\n" + \
               "if you can't detect a digit, press the \"Not a digit\" button"
        tag_question_label = tk.Label(self.newWindow, text=text)
        tag_question_label.config(font=("Arial", 12))
        tag_question_label.pack(pady=10)
        game_image_label.pack()
        digit_entered_str = tk.StringVar(self.newWindow, value=str(suggested_digit))
        jersey_digit_entry = tk.Entry(self.newWindow, width=3, borderwidth=2, relief="groove", font=("Arial", 16),
                                      justify='center',
                                      textvariable=digit_entered_str)
        jersey_digit_entry.pack(pady=6)
        jersey_digit_entry.focus()
        digit_entered_str.trace("w", lambda *args: character_limit(digit_entered_str))

        buttons_frame = tk.LabelFrame(self.newWindow, borderwidth=0)
        buttons_frame.pack(pady=6)
        save_digit_button = tk.Button(buttons_frame, text="Save digit",
                                      command=lambda: self.exit_popup_window(DIGIT_TAGGED,
                                                                             jersey_digit_entry.get(),
                                                                             digit_box_xyxy))
        not_digit_button = tk.Button(buttons_frame, text="Not a digit",
                                     command=lambda: self.exit_popup_window(NO_DIGIT))
        save_digit_button.pack(side='right', padx=8)
        not_digit_button.pack(side='left', padx=8)
        game_image_label.configure(image=bb_image_tk_with_digit)
        game_image_label.image = bb_image_tk_with_digit
        self.newWindow.protocol('WM_DELETE_WINDOW', lambda: self.exit_popup_window(called_by=EXIT_POP_UP))
        self.center(self.newWindow, win_size=(self.W, int((game_image_label.winfo_reqheight() +
                                                          tag_question_label.winfo_reqheight() +
                                                           buttons_frame.winfo_reqheight() +
                                                           jersey_digit_entry.winfo_reqheight() + 42) * 1.1)))

    def exit_popup_window(self, called_by, digit_widget_string=None, digit_box_xyxy=None):
        if called_by == DIGIT_TAGGED:
            if len(digit_widget_string) != 1:
                tk.messagebox.showerror("Error", "You should enter only 1 digit between 0 to 9")
                return
            self.digits_tagged += [(digit_widget_string, digit_box_xyxy)]
            self.newWindow.destroy()
            self.master.deiconify()

        elif called_by == NO_DIGIT:
            self.newWindow.destroy()
        else:          # EXIT_POP_UP
            self.digits_tagged.clear()
            self.newWindow.destroy()
            self.master.deiconify()

    def exit_gui(self):
        self.master.destroy()

    @staticmethod
    def draw_digit_box(bb_img_cv, box_xyxy):
        img = bb_img_cv.copy()
        x1, y1, x2, y2 = box_xyxy
        cv2.rectangle(img, (x1 - 2, y1 - 2), (x2 + 2, y2 + 2), (255, 255, 0), 2)
        cv2.rectangle(img, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
        return img


def read_dataframe(csv_path):
    if not isfile(csv_path + sep + "detections.csv"):
        tk.messagebox.showerror("Manual Player Tagger", "No detections.csv file in this directory")
        raise Exception("Error: No detections.csv file in given directory")
    else:
        csv_filepath = csv_path + sep + "detections.csv"
    df = read_csv(csv_filepath, index_col='bb_id')
    return df, csv_filepath


def get_torso_kp(row):
    return ((row['l_shoulder_x'], row['l_shoulder_y']),
            (row['r_shoulder_x'], row['r_shoulder_y']),
            (row['l_hip_x'], row['l_hip_y']),
            (row['r_hip_x'] ,row['r_hip_y']))


def save_player_jersey_to_df(df, bb_id, digits_tagged):
    df.at[bb_id, 'tagged'] = 1
    for digit_details, digit_index in zip(digits_tagged, ["one", "two"]):
        x1, y1, x2, y2 = digit_details[1]
        df.at[bb_id, 'digit_' + digit_index + '_bb_x'] = x1
        df.at[bb_id, 'digit_' + digit_index + '_bb_y'] = y1
        df.at[bb_id, 'digit_' + digit_index + '_bb_h'] = y2 - y1
        df.at[bb_id, 'digit_' + digit_index + '_bb_w'] = y2 - y1
        df.at[bb_id, 'digit_' + digit_index + '_tag'] = digit_details[0]


def predictions_to_tuples(predictions):
    preds = []
    for i in range(len(predictions[0]['boxes'])):
        preds.append((predictions[0]['labels'][i].tolist(), tuple(predictions[0]['boxes'][i].int().tolist())))
    return preds

def run_gui_on_df(df, csv_path):
    digits_predictor = DigitPredictor(score_thresh=0.7)
    untagged_df = df[df['tagged'].isnull()]
    if len(untagged_df) < 1:
        tk.messagebox.showinfo("Manual Player Tagger", "No more pictures left to tag in this directory")
        return

    for i, (bb_id, row) in enumerate(untagged_df.iterrows()):
        video_id, torso_kp = row['YouTube_ID'], get_torso_kp(row)
        if not isfile(csv_path + sep + video_id + "_" + bb_id + ".jpeg"):
            df.at[bb_id, 'tagged'] = False
        else:
            digits_tagged = []
            bb_image = Image.open(csv_path + sep + video_id + "_" + bb_id + ".jpeg")
            bbs_digits_suggestion = predictions_to_tuples(digits_predictor(bb_image, torso_kp))
            # bbs_digits_suggestion = [(4,(4,5,20,44))]
            if len(bbs_digits_suggestion) == 0:
                # no digits to tag
                df.at[bb_id, 'tagged'] = 0
            else:
                root = tk.Tk()
                app = JerseyTaggerGUI(bb_image, master=root, digits_suggestions=bbs_digits_suggestion,
                                      digits_tagged=digits_tagged,
                                      show_welcome_msg=i < 1)
                app.mainloop()
                res = root.getvar(name='result')

                if res == STOP_TAGGING:
                    break
                elif res:       # number on the jersey was tagged
                    save_player_jersey_to_df(df, bb_id, digits_tagged)
                else:           # not a player or no visible digits
                    df.at[bb_id, 'tagged'] = 0

def main(csv_path):
    df, csv_filepath = read_dataframe(csv_path)
    run_gui_on_df(df, csv_path)
    df.to_csv(csv_filepath, index='bb_id')


def parse_args():
    parser = argparse.ArgumentParser(description="Semi-supervised tagging of players' jersey numbers")
    parser.add_argument("--csv_path", type=str, default=getcwd(),
                        help="Path to the folder that contains the detection.csv file and frames from relevant game")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.csv_path)

# To create exe file with pyinstaller:
# pyinstaller.exe --noconsole --onefile --icon=EM.ico JerseyTaggerGUI.py