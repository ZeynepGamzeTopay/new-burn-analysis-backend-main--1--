using System;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace BurnAnalysisApp.Models
{
    public class VoiceRecording
    {
        [Key]
        public int VoiceRecordingID { get; set; }

        [ForeignKey("ForumPost")]
        [Required] // ForumPostID zorunlu
        public int ForumPostID { get; set; }

        public ForumPost ForumPost { get; set; }

        public string DoctorName { get; set; } // Kaydeden doktorun adı

        public string FilePath { get; set; } // Ses dosyasının yolu (Örn: /uploads/ses_kaydi_1.mp3)

        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    }
}