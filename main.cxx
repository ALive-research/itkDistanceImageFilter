// ITK includes
#include "itkDistanceImageFilter.h"
#include <itkImageFileWriter.h>
#include <itkImageFileReader.h>
#include <itkRealTimeClock.h>

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

    TCLAP::ValueArg<std::string> inputArgument("i", "input", "Input file", true, "None", "string");
    TCLAP::ValueArg<std::string> outputArgument("o", "output", "Output file", true, "None", "string");

    cmd.add(outputArgument);
    cmd.add(inputArgument);

    cmd.parse(argc,argv);

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
  auto realTimeClock = itk::RealTimeClock::New();
  auto distanceImageFilter = DistanceImageFilterType::New();
  distanceImageFilter->SetInput(inputImageReader->GetOutput());
  distanceImageFilter->InsideIsPositiveOff();
  auto start = realTimeClock->GetRealTimeStamp();
  distanceImageFilter->Update();
  auto end = realTimeClock->GetRealTimeStamp();

  std::cout << "Time (ms):" << (end - start).GetTimeInMilliSeconds() << std::endl;

  // =========================================================================
  // Write the output
  // =========================================================================
  auto outputImageWriter = OutputImageWriterType::New();
  outputImageWriter->SetInput(distanceImageFilter->GetOutput());
  outputImageWriter->SetFileName(output);
  outputImageWriter->Write();

  return EXIT_SUCCESS;
}
