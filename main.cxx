// ITK includes
#include "itkDistanceImageFilter.h"
#include <itkImageFileWriter.h>
#include <itkImageFileReader.h>

// TCLAP includes
#include <tclap/ValueArg.h>
#include <tclap/ArgException.h>
#include <tclap/CmdLine.h>

// STD includes
#include <cstdlib>

// ===========================================================================
// Entry point
// ===========================================================================
int main(int argc, char **argv)
{

  // =========================================================================
  // Command-line variables
  // =========================================================================
  std::string input;
  std::string output;

  // =========================================================================
  // Parse arguments
  // =========================================================================
  try
    {
    TCLAP::CmdLine cmd("itkDistanceImageFilter");

    TCLAP::ValueArg<std::string> inputArgument("o", "input", "Input file", true, "None", "string");
    TCLAP::ValueArg<std::string> outputArgument("o", "output", "Output file", true, "None", "string");

    cmd.add(outputArgument);
    cmd.add(inputArgument);

    output = outputArgument.getValue();
    input = inputArgument.getValue();
    }
  catch(TCLAP::ArgException &e)
    {
    std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
    }

  // =========================================================================
  // ITK definitions
  // =========================================================================
  using InputImageType = itk::Image<unsigned short,3>;
  using InputImageReaderType = itk::ImageFileReader<InputImageType>;
  using OutputImageType = itk::Image<float ,3>;
  using OutputImageWriterType = itk::ImageFileWriter<OutputImageType>;
  using DistanceImageFilterType = itk::DistanceImageFilter<InputImageType, OutputImageType>;

  // =========================================================================
  // Read input image
  // =========================================================================
  auto inputImageReader = InputImageReaderType::New();
  inputImageReader->SetFileName(input);
  inputImageReader->Update();

  // =========================================================================
  // Compute distance map
  // =========================================================================
  auto distanceImageFilter = DistanceImageFilterType::New();
  distanceImageFilter->SetInput(inputImageReader->GetOutput());
  distanceImageFilter->Update();

  return EXIT_SUCCESS;
}
