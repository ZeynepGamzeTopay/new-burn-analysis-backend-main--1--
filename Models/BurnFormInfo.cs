namespace BurnAnalysisApp.Models;
    public class BurnFormInfo
    {
        public int Id { get; set; }
        public string ?PatientName { get; set; }
         public string? Email { get; set; } // Hastanın e-posta adresi

        public int Age { get; set; }
        public string ?Gender { get; set; }
        public string ?BurnCause { get; set; }

        public string? BurnArea { get; set; } // Yanık bölgesi

        public DateTime HospitalArrivalDate { get; set; }
        public DateTime BurnOccurrenceDate { get; set; }

         public string? PhotoPath { get; set; }
    }


