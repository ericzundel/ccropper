Using Python 3.11 on Windows from the Microsoft Store

https://github.com/UB-Mannheim/tesseract/wiki

See requirements.txt for packages that we depend on.

Tesseract Tutorials
https://nanonets.com/blog/ocr-with-tesseract/#ocr-with-pytesseract-and-opencv
https://www.projectpro.io/article/how-to-train-tesseract-ocr-python/561


Tesseract Installation
https://linuxhint.com/install-tesseract-windows/

Training Tesseract on a new font
https://towardsdatascience.com/simple-ocr-with-tesseract-a4341e4564b6

https://sourceforge.net/projects/vietocr/files/jTessBoxEditor/

7 segment font at  https://www.keshikan.net/fonts-e.html
DSEG7 Classic Bold seems to match my display.

Train Tesseract directly from the font file in windows with this project
https://github.com/livezingy/tesstrainsh-win

Add to ~/.bash_profile
PATH=$PATH:"/c/Program Files/Tesseract-OCR"
https://github.com/brentvollebregt/auto-py-to-exe

pip install auto-py-to-exe

Disable the dictionaries to get better reading of inventory numbers. E.g.

$ tesseract -c load_system_dawg=false -c  load_freq_dawg=false --psm 7 ./100_0156-lot\ 6.JPG stdout
Lot F6 
https://github.com/brentvollebregt/auto-py-to-exe

pip install auto-py-to-exe