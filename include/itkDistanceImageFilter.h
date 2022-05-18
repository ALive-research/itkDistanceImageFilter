#ifndef itkDistanceImageFilter_h
#define itkDistanceImageFilter_h

#include "itkImageToImageFilter.h"
#include "itkWeightedCentroidKdTreeGenerator.h"
#include "itkListSample.h"

namespace itk
{
template <typename TInputImage, typename TOutputImage>
class ITK_TEMPLATE_EXPORT DistanceImageFilter : public ImageToImageFilter<TInputImage, TOutputImage>
{
public:
  ITK_DISALLOW_COPY_AND_MOVE(DistanceImageFilter);

  /** Extract dimension from input and output image. */
  static constexpr unsigned int InputImageDimension = TInputImage::ImageDimension;
  static constexpr unsigned int OutputImageDimension = TOutputImage::ImageDimension;
  static constexpr unsigned int ImageDimension = TOutputImage::ImageDimension;

  /** Convenient type alias for simplifying declarations. */
  using InputImageType = TInputImage;
  using InputImageConstPointer = typename InputImageType::ConstPointer;

  using OutputImageType = TOutputImage;
  using OutputImagePointer = typename OutputImageType::Pointer;

  /** Standard class type aliases. */
  using Self = DistanceImageFilter;
  using Superclass = ImageToImageFilter<InputImageType, OutputImageType>;

  using Pointer = SmartPointer<Self>;
  using ConstPointer = SmartPointer<const Self>;

  /** Method for creation through the object factory. */
  itkNewMacro(Self);

  /** Runtime information support. */
  itkTypeMacro(DistanceImageFilter, ImageToImageFilter);

  using InputRegionType = typename InputImageType::RegionType;
  using OutputRegionType = typename OutputImageType::RegionType;

  /** Image type alias support */
  using InputPixelType = typename InputImageType::PixelType;
  using OutputPixelType = typename OutputImageType::PixelType;

  using InputSizeType = typename InputImageType::SizeType;
  using InputSizeValueType = typename InputImageType::SizeValueType;
  using OutputSizeType = typename OutputImageType::SizeType;
  using OutputSizeValueType = typename OutputImageType::SizeValueType;

  using InputIndexType = typename InputImageType::IndexType;
  using InputIndexValueType = typename InputImageType::IndexValueType;
  using OutputIndexType = typename OutputImageType::IndexType;
  using OutputIndexValueType = typename OutputImageType::IndexValueType;

  using InputSpacingType = typename InputImageType::SpacingType;
  using OutputSpacingType = typename OutputImageType::SpacingType;
  using OutputImageRegionType = typename OutputImageType::RegionType;

  using MeasurementVectorType = typename itk::Vector<TInputImage>;
  using SampleType = typename itk::Statistics::ListSample<MeasurementVectorType>;
  using TreeGeneratorType = typename itk::Statistics::KdTreeGenerator<SampleType>;

  /** Set if the distance should be squared. */
  itkSetMacro(SquaredDistance, bool);

  /** Get the distance squared. */
  itkGetConstReferenceMacro(SquaredDistance, bool);

  /** Set On/Off if the distance is squared. */
  itkBooleanMacro(SquaredDistance);

  /** Set if the inside represents positive values in the signed distance
   *  map. By convention ON pixels are treated as inside pixels.*/
  itkSetMacro(InsideIsPositive, bool);

  /** Get if the inside represents positive values in the signed distance
   * map. \see GetInsideIsPositive()  */
  itkGetConstReferenceMacro(InsideIsPositive, bool);

  /** Set if the inside represents positive values in the signed distance
   * map. By convention ON pixels are treated as inside pixels. Default is
   * true.                             */
  itkBooleanMacro(InsideIsPositive);

  /** Set if image spacing should be used in computing distances. */
  itkSetMacro(UseImageSpacing, bool);

  /** Get whether spacing is used. */
  itkGetConstReferenceMacro(UseImageSpacing, bool);

  /** Set On/Off whether spacing is used. */
  itkBooleanMacro(UseImageSpacing);

  /**
   * Set the background value which defines the object.  Usually this
   * value is = 0.
   */
  itkSetMacro(BackgroundValue, InputPixelType);
  itkGetConstReferenceMacro(BackgroundValue, InputPixelType);

#ifdef ITK_USE_CONCEPT_CHECKING
  // Begin concept checking
  itkConceptMacro(IntConvertibleToInputCheck, (Concept::Convertible<int, InputPixelType>));
  itkConceptMacro(InputHasNumericTraitsCheck, (Concept::HasNumericTraits<InputPixelType>));
  itkConceptMacro(OutputImagePixelTypeIsFloatingPointCheck, (Concept::IsFloatingPoint<OutputPixelType>));
  // End concept checking
#endif

protected:
  DistanceImageFilter();
  ~DistanceImageFilter() override = default;

  void
  PrintSelf(std::ostream & os, Indent indent) const override;

  void
  GenerateData() override;

  unsigned int
  SplitRequestedRegion(unsigned int i, unsigned int num, OutputImageRegionType & splitRegion) override;

  void
  ThreadedGenerateData(const OutputImageRegionType &, ThreadIdType) override;

  void
  DynamicThreadedGenerateData(const OutputImageRegionType &) override
  {
    itkExceptionMacro("This class requires threadId so it must use classic multi-threading model");
  }

private:
  void
       Voronoi(unsigned int, OutputIndexType idx, OutputImageType * output);
  bool Remove(OutputPixelType, OutputPixelType, OutputPixelType, OutputPixelType, OutputPixelType, OutputPixelType);

  InputPixelType   m_BackgroundValue;
  InputSpacingType m_Spacing;

  unsigned int m_CurrentDimension{ 0 };

  bool m_InsideIsPositive{ false };
  bool m_UseImageSpacing{ true };
  bool m_SquaredDistance{ false };

  const InputImageType * m_InputCache;
  typename TreeGeneratorType::Pointer m_TreeGenerator {nullptr};
};
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#  include "itkDistanceImageFilter.hxx"
#endif

#endif
