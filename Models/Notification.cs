using System;

namespace BurnAnalysisApp.Models
{
    public class Notification
    {
        public int NotificationID { get; set; }
        public int DoctorID { get; set; } // Bildirimi alacak doktorun ID'si
        public string Message { get; set; }
        public int? ForumPostID { get; set; } // İlişkili ForumPost'un ID'si (nullable)
        public ForumPost? ForumPost { get; set; } // İlişki için (nullable)
        public bool IsRead { get; set; } = false; // Varsayılan olarak okunmamış
        public DateTime CreatedAt { get; set; } = DateTime.UtcNow;

    }
}
