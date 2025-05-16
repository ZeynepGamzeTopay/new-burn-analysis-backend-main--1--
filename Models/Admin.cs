using System.ComponentModel.DataAnnotations;

namespace BurnAnalysisApp.Models
{
    public class Admin
    {
        [Key]
        public int AdminID { get; set; }

        // Name artık nullable olabiliyor
        public string? Name { get; set; }

        [Required]
        [EmailAddress] // E-posta formatını doğrulayan bir attribute ekledik
        public string Email { get; set; } = string.Empty;

        [Required]
        [DataType(DataType.Password)] // Şifre verisi için uygun veri türü
        public string Password { get; set; } = string.Empty;
    }
}
