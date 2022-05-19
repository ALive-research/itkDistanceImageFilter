#ifndef itkDistanceImageFilter_hxx
#define itkDistanceImageFilter_hxx

#include "itkDistanceImageFilter.h"

#include <itkImageRegionIteratorWithIndex.h>
#include <itkImageRegionIterator.h>
#include <itkBinaryThresholdImageFilter.h>
#include <itkBinaryContourImageFilter.h>
#include <itkProgressReporter.h>
#include <itkProgressAccumulator.h>
#include <itkMath.h>

#include <vnl/vnl_vector.h>
#include <itkMath.h>

namespace itk
{
template <typename TInputImage, typename TOutputImage>
DistanceImageFilter<TInputImage, TOutputImage>::DistanceImageFilter()
  : m_BackgroundValue(NumericTraits<InputPixelType>::ZeroValue())
  , m_Spacing(0.0)
  , m_InputCache(nullptr)
  , m_TreeGenerator(nullptr)
  , m_Sample(nullptr)
{
  this->DynamicMultiThreadingOff();
  this->m_TreeGenerator = TreeGeneratorType::New();
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

  this->m_Sample = SampleType::New();
  this->m_Sample->SetMeasurementVectorSize(TInputImage::ImageDimension);

  // KD-Tree generation
  typename ConstNeighborhoodIteratorType::RadiusType radius({1,1,1});
  ConstNeighborhoodIteratorType it(radius, m_InputCache, m_InputCache->GetLargestPossibleRegion());

  for(it.GoToBegin(); !it.IsAtEnd(); ++it)
  {
    // The center of the window is not background
    if (it.GetCenterPixel() != m_BackgroundValue)
    {
      for(int i=0; i<27; ++i)
      {
        bool withinBounds;
        auto value = it.GetPixel(i, withinBounds);

        if (withinBounds && it.GetCenterPixel() != value)
        {
          typename InputImageType::PointType point;
          m_InputCache->TransformIndexToPhysicalPoint(it.GetIndex(), point);
          MeasurementVectorType mv;
          mv[0] = point[0]; mv[1] = point[1]; mv[2] = point[2];
          m_Sample->PushBack(mv);
          break;
        }
      }
    }
  }

  m_TreeGenerator->SetSample(m_Sample);
  m_TreeGenerator->SetBucketSize(16);
  m_TreeGenerator->Update();

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

  using TreeType = typename TreeGeneratorType::KdTreeType;
  typename TreeType::Pointer tree = m_TreeGenerator->GetOutput();

  while (!Ot.IsAtEnd())
  {
    itk::Point<float, 3> pointA;
    m_InputCache->TransformIndexToPhysicalPoint(It.GetIndex(), pointA);

    MeasurementVectorType queryPoint;
    queryPoint[0] = pointA[0];
    queryPoint[1] = pointA[1];
    queryPoint[2] = pointA[2];

    typename TreeType::InstanceIdentifierVectorType neighbors;
    unsigned int numNeighbors = 1;
    tree->Search(queryPoint, numNeighbors, neighbors);
    auto measurement = tree->GetMeasurementVector(neighbors[0]);

    itk::Point<float, 3> pointB;
    pointB[0] = measurement[0];
    pointB[1] = measurement[1];
    pointB[2] = measurement[2];
    auto dist = pointA.EuclideanDistanceTo(pointB);

    Ot.Set(dist);

    ++Ot;
    ++It;
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
