using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace BurnAnalysisApp.Models
{
    public class Doctor
    {
        [Key]
        public int DoctorID { get; set; }

        public string? Name { get; set; }

        [Required]
        [EmailAddress]
        public string Email { get; set; } = string.Empty;
        public string Password { get; set; } = string.Empty;

        public bool Verified { get; set; } = false;

        [ForeignKey("Admin")]
        public int? AdminID { get; set; }
        public Admin? Admin { get; set; }
    }
}