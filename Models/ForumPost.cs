using System;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;
using BurnAnalysisApp.Models;

public class ForumPost
{
    [Key]
    public int ForumPostID { get; set; } // Otomatik ID

    [ForeignKey("PatientInfo")]
    public int PatientID { get; set; } // Hangi hastaya ait?

    public PatientInfo Patient { get; set; } // **Hasta bilgisi için ilişki**

    public string DoctorName { get; set; } // Paylaşan doktorun adı
    public string Description { get; set; } // Hasta hakkında notlar
    public string? PhotoPath { get; set; } // Hasta fotoğrafı
    public DateTime CreatedAt { get; set; } = DateTime.UtcNow;
    public virtual ICollection<Notification> Notifications { get; set; } = new List<Notification>();
    public List<Comment> Comments { get; set; } = new List<Comment>(); // Yorumlar
    public List<VoiceRecording> VoiceRecordings { get; set; } = new List<VoiceRecording>(); // Ses Kayıtları
}
