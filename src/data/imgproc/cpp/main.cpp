#include "Binarization.hpp"
#include "Scanner.hpp"
#include "LineSegmentation.hpp"
#include "WordSegmentation.hpp"
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

int main(int argc, char *argv[]) {

    string srcPath = argv[1];
    string outPath = argv[2];

    Mat image = imread(srcPath);

    String name = outPath.substr(outPath.find_last_of("/\\") + 1);
    name = name.substr(0, name.find("."));

    string extension = ".png";
    fs::path linesPath = outPath;
    linesPath /= "lines";

    fs::create_directories(outPath);

    // START Step 1: crop //
    Scanner *scanner = new Scanner();
    Mat imageCropped;
    scanner->process(image, imageCropped);

    fs::path saveCrop = outPath;
    string cropName = name + "_1_crop" + extension;
    saveCrop /= cropName;
    imwrite(saveCrop.u8string(), imageCropped);
    // END Step 1 //


    // START Step 1.1: resize and definitions //
    int newW = 1280;
    int newH = ((newW * imageCropped.rows) / imageCropped.cols);
    resize(imageCropped, imageCropped, Size(newW, newH));

    int chunksNumber = 8;
    int chunksProcess = 4;
    // END Step 1.1 //


    // START Step 2: binarization //
    Binarization *threshold = new Binarization();
    Mat imageBinary, imageSmoothed;
    // default = 0 | otsu = 1 | niblack = 2 | sauvola = 3 | wolf = 4 //
    threshold->binarize(imageCropped, imageBinary, imageSmoothed, true, 1);

    fs::path saveBinary = outPath;
    fs::path saveSmoothed = outPath;
    string binaryName = name + "_2_binary" + extension;
    string smoothedName = name + "_2_smoothed" + extension;
    saveBinary /= binaryName;
    saveSmoothed /= smoothedName;
    imwrite(saveBinary.u8string(), imageBinary);
    imwrite(saveSmoothed.u8string(), imageSmoothed);
    // END Step 2 //


    // START Step 3: line segmentation //
    LineSegmentation *line = new LineSegmentation();
    vector<Mat> lines;
    Mat imageLines = imageBinary.clone(), imageSmoothedLines = imageSmoothed.clone();
    line->segment(imageLines, imageSmoothedLines, lines, chunksNumber, chunksProcess);

    fs::path saveLines = outPath;
    string linesName = name + "_3_lines" + extension;
    saveLines /= linesName;
    imwrite(saveLines.u8string(), imageSmoothedLines);
    fs::create_directories(linesPath);
    for (int i=0; i<lines.size(); i++) {
        string lineIndex = to_string((i+1)*1e-6).substr(5);
        // string wordIndex = lineIndex + "_" + to_string((j+1)*1e-6).substr(5);
        fs::path saveLine = linesPath / (lineIndex + extension);
        imwrite(saveLine.u8string(), lines[i]);
    }
    // END Step 3 //


    // START Step 4: word segmentation //
 //    WordSegmentation *word = new WordSegmentation();
 //    vector<Mat> summary;
 //    word->setKernel(11, 11, 7);


 //    for (int i=0; i<lines.size(); i++) {
 //        string lineIndex = to_string((i+1)*1e-6).substr(5);

 //        vector<Mat> words;
 //        word->segment(lines[i], words);

 //        summary.push_back(words[0]);
 //        words.erase(words.begin());

    //  fs::create_directories(wordsPath);

 //        for (int j=0; j<words.size(); j++) {
 //            string wordIndex = lineIndex + "_" + to_string((j+1)*1e-6).substr(5);
    //      fs::path saveWord = wordsPath / (wordIndex + extension);
 //            imwrite(saveWord.u8string(), words[j]);
 //        }
 //    }

    // fs::path saveSeparateLine = outPath;

    // for (int i=0; i<summary.size(); i++){
    //  string index = "_4_summary_" + to_string((i+1)*1e-6).substr(5);
    //  string separateLineName = name + index + extension;
    //  fs::path saveLine = saveSeparateLine / separateLineName;
 //     imwrite(saveLine.u8string(), summary[i]);
    // }
    // END Step 4 //


    return 0;
}