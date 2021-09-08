import matplotlib as mpl
import seaborn as sns
font_paths = mpl.font_manager.findSystemFonts()
font_objects = mpl.font_manager.createFontList(font_paths)
font_names = [f.name for f in font_objects]
print(font_names)
sns.set_theme(style="whitegrid")
sns.set(rc={"font.size":10,'font.serif':'Times New Roman'})#'font.family':'serif',
logdir='./EJPH/'
figure = ax.get_figure()
figure.savefig('test.png', dpi=500)