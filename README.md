# captcha-crack

Cracking the silly captchas used on American Airlines' in-flight WiFi payment form.

Here is an example of such a captcha:

![Example captcha reading 05366](data/train/05366.jpg)

# What I did

I started by using [a script](data/auto/download.sh) to download 350 captcha images. This was easy: there was a single URL that would always return a new captcha. Then, I hand-labeled 50 training images (`data/train`) and 40 test images (`data/test`); this took about 15 minutes.

After labeling the 90 images, I used k-nearest neighbors to label the remaining 260 images (`data/auto`). To get better accuracy, I cheated: I split each captcha up into 40x50 patches and classified each patch separately. Since kNN had 100% accuracy on the test set, I was fairly confident that the automated labels would be correct.

Once I had all the data labeled, I decided to create a more general solution using a convolutional neural network. The architecture I used is fairly simple; it takes in an image, and outputs a batch of logits (one logit vector per digit). This architecture gets nearly perfect training and test loss, even when only using the 50 training samples that I hand-labeled.
