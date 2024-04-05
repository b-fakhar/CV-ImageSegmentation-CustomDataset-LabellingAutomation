# Image Segmentation Automated ROI Detection Labeling

## Description

This project focuses on developing an algorithm for automatic masking of cartridge case images, specifically targeting firearm-specific features through multi-class semantic segmentation. By leveraging deep learning techniques, the algorithm successfully identifies crucial firearm characteristics such as breech-face impression, aperture shear, and firing pin attributes. This advancement holds significant promise for enhancing firearm identification systems in forensic investigations.

The algorithm's automation streamlines the identification process, offering efficiency and accuracy in detecting firearm features within images. Utilizing the Fadul dataset from the NIST Ballistics Toolmark Research Database, the model was trained and evaluated using a 10-fold cross-validation approach, achieving an average Dice Score of 0.9689. The dataset was split into train and test sets, maintaining fairness in assessing the model's performance on unseen data.

Two training approaches were employed, with the second approach demonstrating superior performance, achieving an average Dice Score of 0.99 and a cross-entropy loss of 0.03. The model's segmentation predictions visually showcase its effectiveness in masking regions of interest within the images.

This proof of concept highlights the potential of deep learning segmentation methods in firearm identification, offering promising results for further advancements in forensic applications.

For a more comprehensive overview of the project, please refer to the  "_Report.pdf_".

## Demo
