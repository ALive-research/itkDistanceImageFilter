// ITK includes
#include "itkDistanceImageFilter.h"
#include <itkImageFileWriter.h>
#include <itkImageFileReader.h>
#include <itkRealTimeClock.h>

// TCLAP includes
#include <tclap/ValueArg.h>
#include <tclap/SwitchArg.h>
#include <tclap/ArgException.h>
#include <tclap/CmdLine.h>

// STD includes
#include <cstdlib>

// =========================================================================
// Arguments structure
// =========================================================================
struct Arguments {
  enum DataType {_short=0, _int};
  std::string inputFileName;
  std::string outputFileName;
  DataType dataType;
  bool isUnsigned;
};

// =========================================================================
// DoIt Lippincott function
// =========================================================================
template <class T> int DoIt(const Arguments &arguments, T)
{
  // =========================================================================
  // ITK definitions
  // =========================================================================
  using InputImageType = itk::Image<T, 3>;
  using InputImageReaderType = itk::ImageFileReader<InputImageType>;
  using OutputImageType = itk::Image<float ,3>;
  using OutputImageWriterType = itk::ImageFileWriter<OutputImageType>;
  using DistanceImageFilterType = itk::DistanceImageFilter<InputImageType, OutputImageType>;

  // =========================================================================
  // Read input image
  // =========================================================================
  auto inputImageReader = InputImageReaderType::New();
  inputImageReader->SetFileName(arguments.inputFileName);
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
  outputImageWriter->SetFileName(arguments.outputFileName);
  outputImageWriter->Write();

  return EXIT_SUCCESS;
}

// ===========================================================================
// Entry point
// ===========================================================================
int main(int argc, char **argv)
{

  Arguments arguments;

  // =========================================================================
  // Parse arguments
  // =========================================================================
  try
    {
    TCLAP::CmdLine cmd("itkDistanceImageFilter");

    TCLAP::ValueArg<std::string> inputArgument("i", "input", "Input file", true, "None", "string");
    TCLAP::ValueArg<std::string> outputArgument("o", "output", "Output file", true, "None", "string");
    TCLAP::ValueArg<unsigned short int> datatypeInput("d", "datatype", "Datatype: (0) short, (1) int" , true, 0, "unsigned short int");
    TCLAP::SwitchArg unsignedInput("u", "unsigned", "Unsigned values", false);

    cmd.add(outputArgument);
    cmd.add(inputArgument);
    cmd.add(unsignedInput);
    cmd.add(datatypeInput);

    cmd.parse(argc,argv);

    arguments.outputFileName = outputArgument.getValue();
    arguments.inputFileName = inputArgument.getValue();
    arguments.dataType = static_cast<Arguments::DataType>(datatypeInput.getValue());
    arguments.isUnsigned = unsignedInput.getValue();

  } catch (TCLAP::ArgException &e) {

    std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
  }

  // =========================================================================
  // Call the right DoIt depending on the input arguments
  // =========================================================================
  switch (arguments.dataType) {
  case Arguments::DataType::_short:
    return DoIt(arguments, arguments.isUnsigned ? static_cast<unsigned short int>(0)
                : static_cast<short int>(0));

  case Arguments::DataType::_int:
    return DoIt(arguments, arguments.isUnsigned ? static_cast<unsigned int>(0)
                : static_cast<int>(0));
  }

  return EXIT_SUCCESS;
}
