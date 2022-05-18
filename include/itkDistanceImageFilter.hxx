/*=========================================================================
 *
 *  Copyright NumFOCUS
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
#ifndef itkDistanceImageFilter_hxx
#define itkDistanceImageFilter_hxx

#include "itkImageRegionIteratorWithIndex.h"
#include "itkImageRegionIterator.h"
#include "itkBinaryThresholdImageFilter.h"
#include "itkBinaryContourImageFilter.h"
#include "itkProgressReporter.h"
#include "itkProgressAccumulator.h"
#include "itkMath.h"
#include "vnl/vnl_vector.h"
#include "itkMath.h"

namespace itk
{
template <typename TInputImage, typename TOutputImage>
DistanceImageFilter<TInputImage, TOutputImage>::DistanceImageFilter()
  : m_BackgroundValue(NumericTraits<InputPixelType>::ZeroValue())
  , m_Spacing(0.0)
  , m_InputCache(nullptr)
{
  this->DynamicMultiThreadingOff();
}

template <typename TInputImage, typename TOutputImage>
unsigned int
DistanceImageFilter<TInputImage, TOutputImage>::SplitRequestedRegion(unsigned int            i,
                                                                                    unsigned int            num,
                                                                                    OutputImageRegionType & splitRegion)
{
  // Get the output pointer
  OutputImageType * outputPtr = this->GetOutput();

  // Initialize the splitRegion to the output requested region
  splitRegion = outputPtr->GetRequestedRegion();

  const OutputSizeType & requestedRegionSize = splitRegion.GetSize();

  OutputIndexType splitIndex = splitRegion.GetIndex();
  OutputSizeType  splitSize = splitRegion.GetSize();

  // split on the outermost dimension available
  // and avoid the current dimension
  int splitAxis = static_cast<int>(outputPtr->GetImageDimension()) - 1;
  while ((requestedRegionSize[splitAxis] == 1) || (splitAxis == static_cast<int>(m_CurrentDimension)))
  {
    --splitAxis;
    if (splitAxis < 0)
    { // cannot split
      itkDebugMacro("Cannot Split");
      return 1;
    }
  }

  // determine the actual number of pieces that will be generated
  auto range = static_cast<double>(requestedRegionSize[splitAxis]);

  auto         valuesPerThread = static_cast<unsigned int>(std::ceil(range / static_cast<double>(num)));
  unsigned int maxThreadIdUsed = static_cast<unsigned int>(std::ceil(range / static_cast<double>(valuesPerThread))) - 1;

  // Split the region
  if (i < maxThreadIdUsed)
  {
    splitIndex[splitAxis] += i * valuesPerThread;
    splitSize[splitAxis] = valuesPerThread;
  }
  if (i == maxThreadIdUsed)
  {
    splitIndex[splitAxis] += i * valuesPerThread;
    // last thread needs to process the "rest" dimension being split
    splitSize[splitAxis] = splitSize[splitAxis] - i * valuesPerThread;
  }

  // set the split region ivars
  splitRegion.SetIndex(splitIndex);
  splitRegion.SetSize(splitSize);

  itkDebugMacro("Split Piece: " << splitRegion);

  return maxThreadIdUsed + 1;
}

template <typename TInputImage, typename TOutputImage>
void
DistanceImageFilter<TInputImage, TOutputImage>::GenerateData()
{
  ThreadIdType numberOfWorkUnits = this->GetNumberOfWorkUnits();

  OutputImageType *      outputPtr = this->GetOutput();
  const InputImageType * inputPtr = this->GetInput();
  m_InputCache = this->GetInput();

  // prepare the data
  this->AllocateOutputs();
  this->m_Spacing = outputPtr->GetSpacing();

  auto progressAcc = ProgressAccumulator::New();
  progressAcc->SetMiniPipelineFilter(this);

  // // compute the boundary of the binary object.
  // // To do that, we erode the binary object. The eroded pixels are the ones
  // // on the boundary. We mark them with the value 2
  // auto binaryFilter = BinaryFilterType::New();

  // binaryFilter->SetLowerThreshold(this->m_BackgroundValue);
  // binaryFilter->SetUpperThreshold(this->m_BackgroundValue);
  // binaryFilter->SetInsideValue(NumericTraits<OutputPixelType>::max());
  // binaryFilter->SetOutsideValue(NumericTraits<OutputPixelType>::ZeroValue());
  // binaryFilter->SetInput(inputPtr);
  // binaryFilter->SetNumberOfWorkUnits(numberOfWorkUnits);
  // progressAcc->RegisterInternalFilter(binaryFilter, 0.1f);
  // binaryFilter->GraftOutput(outputPtr);
  // binaryFilter->Update();

  // // Dilate the inverted image by 1 pixel to give it the same boundary
  // // as the univerted inputPtr.
  // using BorderFilterType = BinaryContourImageFilter<OutputImageType, OutputImageType>;
  // auto borderFilter = BorderFilterType::New();
  // borderFilter->SetInput(binaryFilter->GetOutput());
  // borderFilter->SetForegroundValue(NumericTraits<OutputPixelType>::ZeroValue());
  // borderFilter->SetBackgroundValue(NumericTraits<OutputPixelType>::max());
  // borderFilter->SetFullyConnected(true);
  // borderFilter->SetNumberOfWorkUnits(numberOfWorkUnits);
  // progressAcc->RegisterInternalFilter(borderFilter, 0.23f);
  // borderFilter->Update();

  // this->GraftOutput(borderFilter->GetOutput());

  typename SampleType::Pointer sample = SampleType::New();
  sample->SetMeasurementVectorSize(3);
  typename TreeGeneratorType::Pointer treeGenerator = TreeGeneratorType::New();
  treeGenerator->SetSample(sample);
  treeGenerator->SetBucketSize(16);
  treeGenerator->Update();


  // Set up the multithreaded processing
  typename ImageSource<OutputImageType>::ThreadStruct str;
  str.Filter = this;

  this->GetMultiThreader()->SetNumberOfWorkUnits(numberOfWorkUnits);
  this->GetMultiThreader()->SetSingleMethod(this->ThreaderCallback, &str);

  // multithread the execution
  for (unsigned int d = 0; d < ImageDimension; ++d)
  {
    m_CurrentDimension = d;
    this->GetMultiThreader()->SingleMethodExecute();
  }
}

template <typename TInputImage, typename TOutputImage>
void
DistanceImageFilter<TInputImage, TOutputImage>::ThreadedGenerateData(
  const OutputImageRegionType & outputRegionForThread,
  ThreadIdType                  threadId)
{
  OutputImageType * outputImage = this->GetOutput();

  InputRegionType region = outputRegionForThread;
  InputSizeType   size = region.GetSize();
  InputIndexType  startIndex = outputRegionForThread.GetIndex();

  OutputImageType * outputPtr = this->GetOutput();

  typename OutputImageType::RegionType outputRegion = outputRegionForThread;
  using OutputRealType = typename NumericTraits<OutputPixelType>::RealType;
  using OutputIterator = ImageRegionIterator<OutputImageType>;
  using InputIterator = ImageRegionConstIterator<InputImageType>;

  OutputIterator Ot(outputPtr, outputRegion);
  InputIterator It(m_InputCache, outputRegion);

  Ot.GoToBegin();
  It.GoToBegin();


  while (!Ot.IsAtEnd()) {
    // cast to a real type is required on some platforms
    const auto outputValue = static_cast<OutputPixelType>(
        std::sqrt(static_cast<OutputRealType>(itk::Math::abs(Ot.Get()))));
  }
}


/**
 * Standard "PrintSelf" method
 */
template <typename TInputImage, typename TOutputImage>
void
DistanceImageFilter<TInputImage, TOutputImage>::PrintSelf(std::ostream & os, Indent indent) const
{
  Superclass::PrintSelf(os, indent);
  os << indent << "Background Value: " << this->m_BackgroundValue << std::endl;
  os << indent << "Spacing: " << this->m_Spacing << std::endl;
  os << indent << "Inside is positive: " << this->m_InsideIsPositive << std::endl;
  os << indent << "Use image spacing: " << this->m_UseImageSpacing << std::endl;
  os << indent << "Squared distance: " << this->m_SquaredDistance << std::endl;
}
} // end namespace itk

#endif
