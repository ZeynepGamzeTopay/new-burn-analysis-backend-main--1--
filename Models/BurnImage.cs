using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace BurnAnalysisApp.Models
{
    public class BurnImage
    {
        [Key]
        public int ImageID { get; set; }

        [Required]
        [ForeignKey("Patient")]
        public int PatientID { get; set; }
        public PatientInfo Patient { get; set; } = null!;

        [ForeignKey("Doctor")]
        public int? UploadedBy { get; set; }
        public Doctor? Doctor { get; set; }

        [Required]
        public string FilePath { get; set; } = string.Empty;

        public DateTime UploadDate { get; set; } = DateTime.Now;

        public string? BurnDepth { get; set; }

        public float? SurfaceArea { get; set; }

        public string? BodyLocation { get; set; }

        public float? SeverityScore { get; set; }

        public DateTime? AnalysisDate { get; set; }

        public bool SharedWithDoctors { get; set; } = false;
    }
}
